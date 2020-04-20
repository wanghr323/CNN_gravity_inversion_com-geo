import ctypes
import DataReader as dr
import numpy as np
from ctypes import *

lib_path = "/home/yc/2Dmodels_64/libpyforward.so"

# input_pre_model = '/home/yckj2882/2Dmodels_64/model_one.txt'
# void forward_va(float *Va_a,int xnum,int ynum,int znum,int dx,int dy,int dz,int measureh)
# AplusS(float *Va_a,float *S_a,float *abn_a,int xnum,int ynum,int znum)

def trans_np_as_c_ptr(array, length):
    array = np.reshape(array, length)
    array = np.asarray(array, dtype=np.float32)
    c_array = (ctypes.c_float * len(array))(*array)
    return c_array


def gravity_forward_Va(Nx, Ny, Nz, dx, dy, dz):
    ll = ctypes.cdll.LoadLibrary
    lib = ll(lib_path)
    vanum = (2 * Nx - 1) * (2 * Ny - 1) * Nz
    Va = np.zeros(vanum, dtype=float)
    c_Va = trans_np_as_c_ptr(Va, vanum)
    measure_h = 0.5
    lib.forward_va(c_Va, Nx, Ny, Nz, c_float(dx), c_float(dy), c_float(dz), c_float(measure_h))
    return c_Va


def gravity_forward(model, c_Va, Nx, Ny, Nz):
    ll = ctypes.cdll.LoadLibrary
    lib = ll(lib_path)
    c_model = trans_np_as_c_ptr(model, Nx * Ny * Nz)
    anomal = np.zeros(Nx * Ny, dtype=float)
    c_anomal = trans_np_as_c_ptr(anomal, Nx * Ny)
    lib.AplusS(c_Va, c_model, c_anomal, Nx, Ny, Nz)
    anomal = np.array(c_anomal)
    # anomal = np.reshape(anomal,(Nx,Ny))
    return anomal


