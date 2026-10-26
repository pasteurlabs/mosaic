using TopOpt, Printf
include("/data/solver_unpatched.jl")
# 2x1x1 hex cantilever: nodes on a 3x2x2 lattice, x=0 face clamped, x=2 face loaded
pts = [Float64[x, y, z] for z in 0:1 for y in 0:1 for x in 0:2]
P = permutedims(reduce(hcat, pts))
idx(x, y, z) = x + 3y + 6z   # 0-based
cells = Int64[idx(0,0,0) idx(1,0,0) idx(1,1,0) idx(0,1,0) idx(0,0,1) idx(1,0,1) idx(1,1,1) idx(0,1,1);
              idx(1,0,0) idx(2,0,0) idx(2,1,0) idx(1,1,0) idx(1,0,1) idx(2,0,1) idx(2,1,1) idx(1,1,1)]
dm = Int32[P[i,1] == 0 ? 1 : 0 for i in 1:size(P,1)]
cm = Int32[P[i,1] == 2 ? 1 : 0 for i in 1:size(P,1)]
cv = reshape([0.0, 0.0, -1.0], 1, 3)
f = tempname() * ".inp"
write_inp_hex(f, P, cells, dm, cm, cv, 1.0, 0.3)
println(read(f, String)[1:min(end, 400)])
InpStiffness(f)
