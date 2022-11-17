import fipy as fp
import scipy.special

p = 1.0
dt = 1.0
p_size = 10
c_size = 10
dx = 1.0
dy = 1.0
mesh = fp.Grid2D(dx=dx, dy=dy, nx=p_size, ny=c_size)
mesh = mesh + [[0.001], [-1.0]]
x = mesh.x
y = mesh.y
f = fp.CellVariable(mesh=mesh, value=0.0)

G = scipy.special.erf(x)
conv_x = [-1.0, 0.0] * (y + fp.numerix.sqrt(1.0 + x ** 2))
conv_y = [0.0, -1.0] * ((1.0 - y ** 2) / p)
D_x = [[[1.0, 0.0], [0.0, 0.0]] * G / x]
D_y = [[[0.0, 0.0], [0.0, 1.0]] * (x+y)]

eq = fp.TransientTerm() == (fp.DiffusionTerm(coeff=D_y) +
                            fp.DiffusionTerm(coeff=D_x) +
                            fp.PowerLawConvectionTerm(coeff=conv_x) +
                            fp.PowerLawConvectionTerm(coeff=conv_y))


f.faceGrad.constrain(0, mesh.exteriorFaces)

eq.solve(var=f, dt=dt)