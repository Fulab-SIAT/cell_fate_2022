#%%
import fipy as fp
import numpy as np
from tqdm import tqdm
from pde_solver_cpy import crate_init_p
from joblib import dump
from time import strftime
#%%

''' Function parameters'''
Time = 20
k1, k2 = 10, 10
n1, n2 = 2, 2
D = .1
init_exception =  np.array([20, 20]).reshape(-1, 1)
''' Simulation parameters'''
dt = 1/1800
x1_size = 512
x2_size = 512
Lx1, Lx2 = 1000, 1000
record_step = 10
dx1, dx2 = Lx1 / x1_size, Lx2 / x2_size


mesh = fp.Grid2D(dx=dx1, dy=dx2, nx=x1_size, ny=x2_size)
#%%
x1 = mesh.x
x2 = mesh.y

conv_x = [-1.0, 0.0] * (1./ (1. + (x1/k1)**n1) - x2)
conv_y = [0.0, -1.0] * (1./ (1. + (x2/k2)**n2) - x1)

convection = fp.VanLeerConvectionTerm

eq = fp.TransientTerm() == (fp.DiffusionTerm(coeff=D) +
                            convection(coeff=conv_x) +
                            convection(coeff=conv_y))



z = crate_init_p(mesh.cellCenters.value[0,:], mesh.cellCenters.value[1,:], np.array([[D, 0],[0, D]]), init_exception)
f = fp.CellVariable(mesh=mesh, value=z)
# f.faceGrad.constrain(0, mesh.exteriorFaces)

loop_num = int(Time/dt)
record_size = 0
record_index = []
for i in range(loop_num):
    if i%record_step == 0:
        record_size += 1
        record_index.append(i)

record_time = np.array(record_index) * dt
record_ret = np.empty((x1_size*x2_size, record_size))

j=0
for i in tqdm(range(int(Time/dt))):
    eq.solve(var=f, dt=dt)
    if i%record_step == 0:
        record_ret[..., j] = f.value
        j+=1
record_ret.resize((x1_size, x2_size, record_size))

recordVariables = ['Lx1','Lx2','Time','dt','dx1','dx2','k1','k2','n1','n2','D','record_time', 'record_ret','x1_size','x2_size']

record_dict = dict()
for variable in recordVariables:
    record_dict[variable] = locals()[variable]
timeNow = strftime('%H-%M-%S-%d-%m-%Y')
dump(value=record_dict, filename=f'./fipy_ret_{timeNow}.bin')
    
# %%
