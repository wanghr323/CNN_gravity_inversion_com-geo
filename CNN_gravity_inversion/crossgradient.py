from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import math as math
import tensorflow as tf
from numpy import shape
from PIL import Image
import matplotlib.pyplot as plt


def cross_gradient(m1, m2, dx, dy, dz):
    # didn't use dxdydz yet ,we just think they were 1 now
    Nx, Ny, Nz = m1.shape
    m1dz = m1[1:Nx - 2, 1:Ny - 2, 0:Nz - 3] - m1[1:Nx - 2, 1:Ny - 2, 2:Nz - 1]
    m1dy = m1[1:Nx - 2, 0:Ny - 3, 1:Nz - 2] - m1[1:Nx - 2, 2:Ny - 1, 1:Nz - 2]
    m1dx = m1[0:Nx - 3, 1:Ny - 2, 1:Nz - 2] - m1[2:Nx - 1, 1:Ny - 2, 1:Nz - 2]

    m2dz = m2[1:Nx - 2, 1:Ny - 2, 0:Nz - 3] - m2[1:Nx - 2, 1:Ny - 2, 2:Nz - 1]
    m2dy = m2[1:Nx - 2, 0:Ny - 3, 1:Nz - 2] - m2[1:Nx - 2, 2:Ny - 1, 1:Nz - 2]
    m2dx = m2[0:Nx - 3, 1:Ny - 2, 1:Nz - 2] - m2[2:Nx - 1, 1:Ny - 2, 1:Nz - 2]

    tx = (m2dz * m1dy - m2dy * m1dz) / 4
    ty = (m2dx * m1dz - m2dz * m1dx) / 4
    tz = (m2dy * m1dx - m2dx * m1dy) / 4

    cg = math.sqrt(np.linalg.norm(tx, ord=None, axis=None, keepdims=False)**2 + \
         np.linalg.norm(ty, ord=None, axis=None, keepdims=False)**2 + \
         np.linalg.norm(tz, ord=None, axis=None, keepdims=False)**2)
    return cg
