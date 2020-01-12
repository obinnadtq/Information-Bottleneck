import time
from mapper_class import Mapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(10)

SNR_db = 6  # SNR in db
Nx = 4  # cardinality of source signal
Ny = 256  # cardinality of quantizer input
Nz = np.arange(1, 65, 1)  # cardinality of quantizer output
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
# Iterative IB
beta4 = 5
convergence_param = 10 ** -4
Ixz = []
Iyz = []
LG = []
count = 0
counter = 0

for j in range(len(Nz)):
    pz_y = np.zeros((Nz[j], Ny))
    for i in range(0, Nz[j]):
        temp = np.arange((i * np.floor(Ny / Nz[j])), min(((i + 1) * np.floor(Ny / Nz[j])), Ny), dtype=int)
        pz_y[i, temp] = 1
    if temp[-1] < Ny:
        pz_y[i, temp[-1] + 1:] = 1
    p_z = np.sum(np.tile(p_y, (Nz[j], 1)) * pz_y, 1)
    pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
    p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz[j], 1, 1))  # p(x,y) expanded dimension
    px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, 4)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
    px_z_expanded = np.tile(np.expand_dims(px_z, axis=1), (1, Ny, 1))
    px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz[j], 1, 1))
    I_x_y = np.sum(
        p_x_y * (np.log(p_x_y) - np.tile(np.expand_dims(np.log(np.tile(p_x, Ny // Nx) * p_y), axis=1), (1, Nx))))
    H_y = -np.sum(p_y * np.log(p_y))
    while True:
        KL = np.sum((np.log(px_y_expanded + 1e-31) - np.log(px_z_expanded + 1e-31)) * px_y_expanded, 2)  # KL divergence
        exponential_term = np.exp(-(beta4 * KL))
        numerator = np.tile(np.expand_dims(p_z, axis=1), (1, Ny)) * exponential_term
        denominator = np.sum(numerator, 0)
        pz_y1 = numerator / denominator
        p_z1 = np.sum(p_y * pz_y1, 1)
        g = np.tile(np.expand_dims(pz_y1, axis=2), (1, 1, Nx)) * p_x_y_expanded
        g1 = np.sum(g, 1) / np.tile(np.expand_dims(p_z1 + 1e-31, axis=1), (1, 4))
        px_z1 = np.nan_to_num(g1)
        pi = [0.5, 0.5]
        p = pi[0] * pz_y1 + pi[1] * pz_y
        KL1 = np.sum((np.log(pz_y1 + 1e-31) - np.log(p + 1e-31)) * pz_y1)
        KL2 = np.sum((np.log(pz_y + 1e-31) - np.log(p + 1e-31)) * pz_y)
        JS = pi[0] * KL1 + pi[1] * KL2
        if JS <= convergence_param:
            p_x_z = px_z1 * np.tile(np.expand_dims(p_z1, axis=1), (1, Nx))
            w = np.tile(np.expand_dims(p_x, axis=0), (Nz[j], 1)) * np.tile(np.expand_dims(p_z1, axis=1), (1, Nx))
            w1 = np.log(p_x_z + 1e-31) - np.log(w + 1e-31)
            I_x_z = np.sum(p_x_z * w1)
            p_y_z = pz_y1 * np.tile(np.expand_dims(p_y, axis=0), (Nz[j], 1))
            w2 = np.tile(np.expand_dims(p_y, axis=0), (Nz[j], 1)) * np.tile(np.expand_dims(p_z1, axis=1), (1, Ny))
            w3 = np.log(p_y_z + 1e-31) - np.log(w2 + 1e-31)
            I_y_z = np.sum(p_y_z * w3)
            L = I_y_z - beta4 * I_x_z
            Ixz.append(I_x_z)
            Iyz.append(I_y_z)
            LG.append(L)
            break
        else:
            p_z = p_z1
            pz_y = pz_y1
        count = count + 1
    counter = counter + 1
plt.plot(Nz, LG, 'g', linewidth=2)
plt.title('Number of Clusters vs Lagrangian Function')
plt.xlabel('Number of Clusters')
plt.ylabel('L[p(z|y)]')
plt.grid()
plt.show()

