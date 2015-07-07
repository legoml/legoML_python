from numpy import hstack, squeeze
from numpy.random import randint, randn
from scipy.stats import multivariate_normal
from sympy import log
from sympy.matrices import Matrix, MatrixSymbol
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import gaussian_density_function,\
    one_density_function

max_num_dim = 3
max_abs_value = 3

nx_value = randint(max_num_dim) + 1
ny_value = randint(max_num_dim) + 1
nz_value = randint(max_num_dim) + 1
nw_value = randint(max_num_dim) + 1

mx_value = max_abs_value * randn(1, nx_value)
my_value = max_abs_value * randn(1, ny_value)
mz_value = max_abs_value * randn(1, nz_value)
mw_value = max_abs_value * randn(1, nw_value)
M_value = hstack((mx_value, my_value, mz_value, mw_value))

r_value = randn(nx_value + ny_value + nz_value + nw_value, 4 * max_num_dim)
S_value = r_value.dot(r_value.T)
Sx_value = S_value[:nx_value,
                   :nx_value]
Sy_value = S_value[nx_value:(nx_value + ny_value),
                   nx_value:(nx_value + ny_value)]
Sz_value = S_value[(nx_value + ny_value):(nx_value + ny_value + nz_value),
                   (nx_value + ny_value):(nx_value + ny_value + nz_value)]
Sw_value = S_value[(nx_value + ny_value + nz_value):(nx_value + ny_value + nz_value + nw_value),
                   (nx_value + ny_value + nz_value):(nx_value + ny_value + nz_value + nw_value)]
Sxy_value = S_value[:nx_value,
                    nx_value:(nx_value + ny_value)]
Sxz_value = S_value[:nx_value,
                    (nx_value + ny_value):(nx_value + ny_value + nz_value)]
Sxw_value = S_value[:nx_value,
                    (nx_value + ny_value + nz_value):(nx_value + ny_value + nz_value + nw_value)]
Syz_value = S_value[nx_value:(nx_value + ny_value),
                    (nx_value + ny_value):(nx_value + ny_value + nz_value)]
Syw_value = S_value[nx_value:(nx_value + ny_value),
                    (nx_value + ny_value + nz_value):(nx_value + ny_value + nz_value + nw_value)]
Szw_value = S_value[(nx_value + ny_value):(nx_value + ny_value + nz_value),
                     (nx_value + ny_value + nz_value):(nx_value + ny_value + nz_value + nw_value)]

x_value = max_abs_value * randn(1, nx_value)
y_value = max_abs_value * randn(1, ny_value)
z_value = max_abs_value * randn(1, nz_value)
w_value = max_abs_value * randn(1, nw_value)
X_value = hstack((x_value, y_value, z_value, w_value))

X = multivariate_normal(mean=squeeze(M_value).tolist(),
                        cov=S_value.tolist())
print(-log(X.pdf(squeeze(X_value.T).tolist())))


nx, ny, nz, nw = nx_value, ny_value, nz_value, nw_value
x = MatrixSymbol('x', 1, nx)
y = MatrixSymbol('y', 1, ny)
z = MatrixSymbol('z', 1, nz)
w = MatrixSymbol('w', 1, nw)
mx = MatrixSymbol('mx', 1, nx)
my = MatrixSymbol('my', 1, ny)
mz = MatrixSymbol('mz', 1, nz)
mw = MatrixSymbol('mw', 1, nw)
Sx = MatrixSymbol('Sx', nx, nx)
Sy = MatrixSymbol('Sy', ny, ny)
Sz = MatrixSymbol('Sz', nz, nz)
Sw = MatrixSymbol('Sw', nw, nw)
Sxy = MatrixSymbol('Sxy', nx, ny)
Sxz = MatrixSymbol('Sxz', nx, nz)
Sxw = MatrixSymbol('Sxw', nx, nw)
Syz = MatrixSymbol('Syz', ny, nz)
Syw = MatrixSymbol('Syw', ny, nw)
Szw = MatrixSymbol('Szw', nz, nw)

g = gaussian_density_function(dict(x=x, y=y, z=z, w=w),
                              {('mean', 'x'): mx, ('mean', 'y'): my, ('mean', 'z'): mz, ('mean', 'w'): mw,
                               ('cov', 'x'): Sx, ('cov', 'y'): Sy, ('cov', 'z'): Sz, ('cov', 'w'): Sw,
                               ('cov', 'x', 'y'): Sxy, ('cov', 'x', 'z'): Sxz, ('cov', 'x', 'w'): Sxw,
                               ('cov', 'y', 'z'): Syz, ('cov', 'y', 'w'): Syw, ('cov', 'z', 'w'): Szw})
g.pprint()

p = g({'x': Matrix(x_value), 'y': Matrix(y_value), 'z': Matrix(z_value), 'w': Matrix(w_value),
       ('mean', 'x'): Matrix(mx_value), ('mean', 'y'): Matrix(my_value),
       ('mean', 'z'): Matrix(mz_value), ('mean', 'w'): Matrix(mw_value),
       ('cov', 'x'): Matrix(Sx_value), ('cov', 'y'): Matrix(Sy_value),
       ('cov', 'z'): Matrix(Sz_value), ('cov', 'w'): Matrix(Sw_value),
       ('cov', 'x', 'y'): Matrix(Sxy_value), ('cov', 'x', 'z'): Matrix(Sxz_value),
       ('cov', 'x', 'w'): Matrix(Sxw_value), ('cov', 'y', 'z'): Matrix(Syz_value),
       ('cov', 'y', 'w'): Matrix(Syw_value), ('cov', 'z', 'w'): Matrix(Szw_value)}, return_probability=False)
print(p)

g_a = g.at(dict(w=Matrix(w_value)))
g_a.pprint()

g_max = g.max()
g_max.pprint()

g_a_max = g_a.max()
g_a_max.pprint()

g_m_wz = g.marginalize(('w', 'z'))
g_m_wz.pprint()

g_c = g.condition(dict(w=Matrix(w_value)))
g_c.pprint()

g_one = g_m = g.marginalize(('x', 'y', 'z', 'w'))
g_one.pprint()

f = g.multiply(one_density_function())