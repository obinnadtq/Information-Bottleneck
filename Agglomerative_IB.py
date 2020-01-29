import time
from mapper_class import Mapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(10)
# parameters
SNR_db = 6  # SNR in db
Nx = 4  # cardinality of source signal
Ny = 64# cardinality of quantizer input
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
# Agglomerative IB
# initialization
beta1 = np.arange(0.01, 5, 0.001)
beta2 = np.arange(5, 400, 0.1)
beta = np.concatenate((beta1, beta2), axis=None)
Ixz = []
Iyz = []
for k in range(len(beta)):
    Nz = Ny
    pz_y = np.zeros((Nz, Ny))
    for i in range(0, Nz):
        temp = np.arange((i * np.floor(Ny / Nz)), min(((i + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
        pz_y[i, temp] = 1
    if temp[-1] < Ny:
        pz_y[i, temp[-1] + 1:] = 1
    while Nz > 32:
        p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
        pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
        p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))  # p(x,y) expanded dimension
        px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, 4)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
        I_x_y = np.sum(
            p_x_y * (np.log(p_x_y) - np.tile(np.expand_dims(np.log(np.tile(p_x, Ny // Nx) * p_y), axis=1), (1, Nx))))
        H_y = -np.sum(p_y * np.log(p_y))
        minValue = 1
        for i in range(len(pz_y)):
            count = 0
            while count < Nz:
                pz_y_updated = np.copy(pz_y)
                if count != i:
                    pz_y_new = pz_y[count] + pz_y[i]
                    pz_y_updated[count] = pz_y_new
                    pz_y_updated = np.delete(pz_y_updated, i, axis=0)
                    pz_y_updated = pz_y_updated
                    p_z_new = p_z[count] + p_z[i]
                    pi = [p_z[count] / p_z_new, p_z[i] / p_z_new]
                    p_bar1 = pi[0] * px_z[count] + pi[1] * px_z[i]
                    DKL1 = np.sum(px_z[count] * (np.log(px_z[count]) - np.log(p_bar1)))
                    DKL2 = np.sum(px_z[i] * (np.log(px_z[i]) - np.log(p_bar1)))
                    JS_1_2 = pi[0] * DKL1 + pi[1] * DKL2
                    py_z_i = pz_y[count] * p_y / p_z[count]
                    py_z_j = pz_y[i] * p_y / p_z[i]
                    p_bar2 = pi[0] * py_z_i + pi[1] * py_z_j
                    DKL3 = np.sum(py_z_i * (np.log(py_z_i + 1e-31) - np.log(p_bar2 + 1e-31)))
                    DKL4 = np.sum(py_z_j * (np.log(py_z_j + 1e-31 - np.log(p_bar2 + 1e-31))))
                    JS_3_4 = pi[0] * DKL3 + pi[1] * DKL4
                    dist = JS_1_2 - (1 / beta[k]) * JS_3_4
                    deltaLMax = p_z_new * dist
                    if deltaLMax < minValue:
                        minValue = deltaLMax
                        best = pz_y_updated
                count = count + 1
        pz_y = best
        Nz = Nz - 1
        if Nz == 32:
            p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
            pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
            p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))  # p(x,y) expanded dimension
            px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, 4)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
            p_x_z = px_z * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
            w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
            w1 = np.log(p_x_z + 1e-31) - np.log(w + 1e-31)
            I_x_z = np.sum(p_x_z * w1)
            p_y_z = pz_y * np.tile(np.expand_dims(p_y, axis=0), (Nz, 1))
            w2 = np.tile(np.expand_dims(p_y, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Ny))
            w3 = np.log(p_y_z + 1e-31) - np.log(w2 + 1e-31)
            I_y_z = np.sum(p_y_z * w3)
            Ixz.append(I_x_z)
            Iyz.append(I_y_z)
plt.plot(Iyz, Ixz, 'b+-', linewidth=2)
plt.title('Relevant-Compression Information Plot for Agglomerative Information Bottleneck')
plt.xlabel('I(Z;Y)')
plt.ylabel('I(Z;X)')
plt.show()