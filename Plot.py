#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 01:20:00 2020

@author: obinnaisiwekpeni
"""
import matplotlib.pyplot as plt
import numpy as np
H_y = 3.55443706652109
I_x_y = 1.0948890517615104
Agglo_IXZ = []
Agglo_IYZ = []
IIB_IXZ = []
IIB_IYZ = []
Norm_Agglo_IXZ = []
Norm_Agglo_IYZ = []
Norm_IIB_IXZ = []
Norm_IIB_IYZ = []
with open("Agglo_IXZ.txt") as file1:
    lines = file1.readlines()
    Agglo_XZ = [line.split()[0] for line in lines]

with open("Agglo_IYZ.txt") as file2:
    lines = file2.readlines()
    Agglo_YZ = [line.split()[0] for line in lines]

with open("IIB_IXZ.txt") as file1:
    lines = file1.readlines()
    IIB_XZ = [line.split()[0] for line in lines]

with open("IIB_IYZ.txt") as file2:
    lines = file2.readlines()
    IIB_YZ = [line.split()[0] for line in lines]

for i in Agglo_XZ:
    Agglo_IXZ.append(float(i))

for i in Agglo_YZ:
    Agglo_IYZ.append(float(i))

for i in IIB_XZ:
    IIB_IXZ.append(float(i))

for i in IIB_YZ:
    IIB_IYZ.append(float(i))

for i in Agglo_IXZ:
    Norm_Agglo_IXZ.append(i/I_x_y)

for i in IIB_IXZ:
    Norm_IIB_IXZ.append(i/I_x_y)

for i in Agglo_IYZ:
    Norm_Agglo_IYZ.append(i/H_y)

for i in IIB_IYZ:
    Norm_IIB_IYZ.append(i/H_y)

plt.figure()
ax1 = plt.axes()
ax1.set_facecolor('lightgrey')
plt.plot(Agglo_IYZ, Agglo_IXZ, 'b+-', linewidth=2, label='Agglomerative IB')
plt.plot(IIB_IYZ, IIB_IXZ, 'r+-', linewidth=2, label = 'Iterative IB')
plt.legend()
plt.grid()
plt.title('Relevance-Compression Plot for  Iterative IB and Agglomerative IB')
plt.xlabel('I(Y;Z)')
plt.ylabel('I(X;Z)')

plt.figure()
ax2 = plt.axes()
ax2.set_facecolor('lightgrey')
plt.plot(Norm_Agglo_IYZ, Norm_Agglo_IXZ, 'b+-', linewidth=2, label='Agglomerative IB')
plt.plot(Norm_IIB_IYZ, Norm_IIB_IXZ, 'r+-', linewidth=2, label='Iterative IB')
plt.legend()
plt.title('Normalized Relevance-Compression Plot for  Iterative IB and Agglomerative IB')
plt.grid()
plt.xlabel('Normalized I(Y;Z)')
plt.ylabel('Normalized I(X;Z)')
plt.show()
