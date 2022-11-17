#%%
from fipy import *
#numerix.random.seed(13)
from scipy import interpolate
import matplotlib.pyplot as plt
# import matplotlib.mlab as ml
from pde_solver_cpy import crate_init_p
import time
import numpy as np
from tqdm import tqdm
fig = plt.figure(figsize=(15,6))
ax1 = plt.subplot((121))
ax2 = plt.subplot((122))

# plt.ion()
# plt.show()

''' Landscape parameters '''
L = 10.
nxy = 512
dxy = L/nxy
m = Grid2D(nx=nxy, ny=nxy, dx=dxy, dy=dxy)
ulist = m.x.value[:nxy]

''' Deterrent signals '''
ido = numerix.random.randint(nxy, size=(100,2))
# -- interpolate points
og = numerix.zeros((nxy, nxy))
numerix.add.at(og, (ido[:,0], ido[:,1]), 1)
signal = CellVariable(mesh=m, value=og.flat)

''' Convection-Diffusion kernel '''
# -- Parameters
D = 1.
c = 1.
b = ((c,),(c,))
delta = 2./(dxy * 0.5) # smoothing parameter 
convection = VanLeerConvectionTerm

# -- Initialization
attractor = (3., 5.)
# z = ml.bivariate_normal(m.x.value, m.y, .1, .1, attractor[0], attractor[1])
z = crate_init_p(m.x.value, m.y.value, np.array([[1, 0],[0, 1]]), np.array([1, 1]).reshape(-1, 1))
var = CellVariable(mesh=m, value=z)

# -- Spatial objects assignments
s = m.faceCenters
r = (s - attractor).mag # radius of any position to the attractor
faceVelocity = c*numerix.tanh(delta*r)*(s-attractor)/r  # convection component in the absence of signal

# -- Fokker-Planck equation
eq = (TransientTerm() == DiffusionTerm(coeff=D) + convection(coeff=faceVelocity * signal.faceValue))
#%%
# -- Visualization
if __name__ == '__main__':
    viewer = Matplotlib2DGridContourViewer(vars=var,
    limits = {'xmin': 0, 'xmax': L, 'ymin': 0, 'ymax': L},
    cmap = plt.cm.Greens,
    axes = ax1)
    
    velocityViewer = MatplotlibStreamViewer(vars=faceVelocity * signal.faceValue, 
                                            color=(faceVelocity * signal.faceValue).mag,
                                            xmin=0, xmax=L, ymin=0, ymax=L,
                                            axes=ax2)
    ax2.plot(ulist[ido[:,1]], ulist[ido[:,0]], 'ko', alpha=.3)
    plt.draw()


''' Simulation '''
dt = 0.1 * 1./2 * dxy
time = 10
for step in tqdm(range(int(time/dt))):
    # print( 'time step', step)
    eq.solve(var=var, dt = dt)
    print( 'cell volume: ', var.cellVolumeAverage() * m.cellVolumes.sum())
    viewer.plot()
# %%
