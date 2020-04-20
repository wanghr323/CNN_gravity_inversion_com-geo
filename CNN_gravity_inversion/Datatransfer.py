from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def anomal_tran_pos(anomal, model_max):
    return (1/model_max) * anomal


def anomal_tran_neg(anomal, model_min):
    return (-1/model_min) * anomal


def anomal_tran_deltz(anomal, dz):
    print('shape:\n', np.shape(anomal))

    result = ((1/dz)) * anomal
    print('shape:\n', np.shape(result))
    return result