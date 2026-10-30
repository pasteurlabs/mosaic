"""JAX state-only custom VJP over unchanged native API in isolated process."""
import atexit, json, os, subprocess, sys, threading, time
from pathlib import Path
from multiprocessing.connection import Client
import jax
import numpy as np
class NativeBridge:
 def __init__(self,api):
  socket=Path(os.environ['WORKDIR'])/'native-api.sock'
  self.process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('native_worker.py')),'--api',str(api),'--socket',str(socket)])
  for _ in range(600):
   if socket.exists():break
   if self.process.poll() is not None:raise RuntimeError('Native worker exited before readiness')
   time.sleep(1)
  else:raise TimeoutError('Native worker initialization exceeded600s')
  self.connection=Client(str(socket),family='AF_UNIX',authkey=b'local-native-api');self.lock=threading.Lock();self.cache={};atexit.register(self.close)
 def close(self):
  try:self.connection.send({'op':'stop'});self.connection.close()
  except (EOFError,BrokenPipeError,OSError):pass
  try:self.process.wait(timeout=10)
  except subprocess.TimeoutExpired:self.process.terminate()
 def request(self,op,inputs,cotangent=None):
  with self.lock:
   self.connection.send({'op':op,'inputs':inputs,'cotangent':cotangent});response=self.connection.recv()
  if not response['ok']:raise RuntimeError(response['traceback'])
  return response['result']
 def apply(self,t,inputs):
  state=inputs['state0'];fixed={k:v for k,v in inputs.items() if k!='state0'}
  key=json.dumps({k:np.asarray(v).tolist() for k,v in fixed.items()},sort_keys=True)
  if key not in self.cache:
   def forward(x):
    shape=jax.ShapeDtypeStruct(x.shape,x.dtype)
    return jax.pure_callback(lambda a:np.asarray(self.request('apply',{**fixed,'state0':a})['result'],dtype=a.dtype),shape,x)
   @jax.custom_vjp
   def advance(x):return forward(x)
   def fwd(x):return forward(x),x
   def bwd(x,cotangent):
    shape=jax.ShapeDtypeStruct(x.shape,x.dtype)
    gradient=jax.pure_callback(lambda a,q:np.asarray(self.request('vjp',{**fixed,'state0':a},q)['state0'],dtype=a.dtype),shape,x,cotangent)
    return (gradient,)
   advance.defvjp(fwd,bwd);self.cache[key]=advance
  return {'result':self.cache[key](state)}
