import time
from mapper_class import Mapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(10)

# parameters
SNR_db = 6  # SNR in db
Nx = 4  # cardinality of source signal
Ny = 8  # cardinality of quantizer input
Nz = 8  # cardinality of quantizer output
alphabet = np.array([-1.5, -0.5, 0.5, 1.5])

# mapping to ASK
u = np.random.randint(2, size=40)  # randomly generated bits
mapper = Mapper()
mapper.generate_mapping(Nx, 'ASK')
# mapper.show_mapping()
x = mapper.map_symbol(u)
x = x[0]

# AWGN Channel
SNR_lin = 10 ** (SNR_db / 10)
sigma2_X = 1
sigma2_N = sigma2_X / SNR_lin
sigma_N_2 = np.sqrt(sigma2_N / 2)
n = (np.random.randn(np.size(x)) + 1j * np.random.randn(np.size(x))) * sigma_N_2

# channel output
y = x + n

# sampling interval
dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(sigma2_N)) / Ny

# discrete y
y_d = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(sigma2_N)), (np.amax(alphabet) + 5 * np.sqrt(sigma2_N)), dy).reshape(
    Ny, 1)

p_x = 1 / Nx * np.ones(Nx)  # p(x)
py_x = np.ones((Ny, Nx))  # initialized p(y|x)
tmp = np.zeros((Ny, 1))  # initialize the temporary matrix to store conditional pdf

for i in range(0, Nx):
    v = np.exp(-(y_d - alphabet[i]) ** 2 / sigma2_N)
    tmp = np.append(tmp, v, axis=1)

py = tmp[:, -Nx:]  # remove the zero from the first column of the matrix
py_x = py_x * py  # obtain the conditional pdf

norm_sum = np.sum(py_x, 0)  # normalization for the pdf
py_x = py_x / np.tile(norm_sum, (Ny, 1))  # p(y|x)
p_x_y = py_x * p_x  # p(x,y) joint probability of x and y
p_y = np.sum(p_x_y, 1)  # p(y)
px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx) / p_y, axis=1), (1, Nx))

# initialization of p(z|y)
pz_y = np.zeros((Nz, Ny))
for i in range(0, Nz):
    temp = np.arange((i * np.floor(Ny / Nz)), min(((i + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
    pz_y[i, temp] = 1
if temp[-1] < Ny:
    pz_y[i, temp[-1] + 1:] = 1
p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))  # p(x,y) expanded dimension
px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, 4)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
px_z_expanded = np.tile(np.expand_dims(px_z, axis=1), (1, Ny, 1))
px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
I_x_y = np.sum(p_x_y * (np.log(p_x_y) - np.tile(np.expand_dims(np.log(np.tile(p_x, Ny // Nx) * p_y), axis=1), (1, Nx))))
H_y = -np.sum(p_y * np.log(p_y))
Ixz = []
Iyz = []

# Agglomerative IB
beta = 5
# for i in range(len(pz_y)):
pz_y_new = pz_y[0] + pz_y[1]
pz_y_updated = np.copy(pz_y)
pz_y_updated[0] = pz_y_new
pz_y_updated = np.delete(pz_y_updated, 1, axis=0)
p_z_new = p_z[0] + p_z[1]
pi = [p_z[0] / p_z_new, p_z[1] / p_z_new]
p_bar1 = pi[0] * px_z[0] + pi[1] * px_z[1]
DKL1 = np.sum(px_z[0] * (np.log(px_z[0] - np.log(p_bar1))))
DKL2 = np.sum(px_z[2] * (np.log(px_z[1] - np.log(p_bar1))))
JS_1_2 = pi[0] * DKL1 + pi[1] * DKL2
py_z_i = pz_y[0] * p_y/p_z[0]
py_z_j = pz_y[1] * p_y/p_z[1]
p_bar2 = pi[0] * py_z_i + pi[1] * py_z_j
DKL3 = np.sum(py_z_i * (np.log(py_z_i+1e-31) - np.log(p_bar2+1e-31)))
DKL4 = np.sum(py_z_j * (np.log(py_z_j+1e-31 - np.log(p_bar2+1e-31))))
JS_3_4 = pi[0] * DKL3 + pi[1] * DKL4
dist = JS_1_2 - (1/beta) * JS_3_4
deltaLMax = p_z_new * dist
obinna = 5
