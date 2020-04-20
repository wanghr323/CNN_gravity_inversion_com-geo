from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt


def plot_model(d_model_t, title = 'none'):
    slice_t_1 = d_model_t[1, :, :].T
    slice_t_2 = d_model_t[2, :, :].T
    slice_t_3 = d_model_t[3, :, :].T
    slice_t_4 = d_model_t[4, :, :].T
    slice_t_5 = d_model_t[5, :, :].T

    slice_t_6 = d_model_t[6, :, :].T
    slice_t_7 = d_model_t[7, :, :].T
    slice_t_8 = d_model_t[8, :, :].T
    slice_t_9 = d_model_t[9, :, :].T

    vmin_t = (d_model_t[:, :, :]).min()
    vmax_t = (d_model_t[:, :, :]).max()

    plt.subplot(5, 5, 1)
    plt.pcolor(slice_t_1, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 2)
    plt.pcolor(slice_t_2, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 3)
    plt.pcolor(slice_t_3, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 4)
    plt.pcolor(slice_t_4, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 5)
    plt.pcolor(slice_t_5, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 6)
    plt.pcolor(slice_t_6, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 7)
    plt.pcolor(slice_t_7, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 8)
    plt.pcolor(slice_t_8, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()

    plt.subplot(5, 5, 9)
    plt.pcolor(slice_t_9, vmin=vmin_t, vmax=vmax_t)
    plt.colorbar()
    plt.title(title, fontsize='large',fontweight = 'bold')
    plt.show()


def plot_model_64(d_model_t, title = 'none'):
    Nz ,Nx, Ny = d_model_t.shape
    vmin_t = (d_model_t[:, :, :]).min()
    vmax_t = (d_model_t[:, :, :]).max()
    if Nz > 25:
        Nz = 25

    for i in range(1, Nz+1):
        slice_t = d_model_t[i-1, :, :].T
        plt.subplot(5, 5, i)
        plt.pcolor(slice_t, vmin=vmin_t, vmax=vmax_t)
        plt.colorbar()

    plt.title(title, fontsize='large', fontweight='bold')
    plt.show()


def plot_anomal(anomal, title = 'none'):
    plt.title(title, fontsize='large', fontweight='bold')
    plt.pcolor(anomal)
    plt.colorbar()
    plt.show()


def history_plot(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()