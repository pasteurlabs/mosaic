"""Single-thread-owned Julia API worker; external process isolation only."""
import argparse, importlib.util, traceback
from multiprocessing.connection import Listener
p=argparse.ArgumentParser();p.add_argument('--api');p.add_argument('--socket');a=p.parse_args()
spec=importlib.util.spec_from_file_location('isolated_native',a.api);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
with Listener(a.socket,family='AF_UNIX',authkey=b'local-native-api') as listener:
 with listener.accept() as connection:
  while True:
   try:request=connection.recv()
   except EOFError:break
   if request['op']=='stop':break
   try:
    inputs=native.InputSchema(**request['inputs'])
    if request['op']=='apply':result=native.apply(inputs)
    elif request['op']=='vjp':result=native.vector_jacobian_product(inputs,{'state0'},{'result'},{'result':request['cotangent']})
    else:raise ValueError(request['op'])
    connection.send({'ok':True,'result':result})
   except Exception as error:connection.send({'ok':False,'error':str(error),'traceback':traceback.format_exc()})
