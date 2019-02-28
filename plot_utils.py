import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def plot_eigenvalues(eigenvalues):
    plt.figure(figsize=(8,8))
    plt.gcf()
    ax = plt.gca()

    points = ax.plot(
            eigenvalues.real, eigenvalues.imag, 'bo', label='Eigenvalues')

    # set limits for axis
    limit = np.max(np.ceil(np.absolute(eigenvalues)))
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))

    plt.ylabel('Imaginary part')
    plt.xlabel('Real part')

    unit_circle = plt.Circle(
        (0., 0.),
        1.,
        color='green',
        fill=False,
        label='Unit circle',
        linestyle='--')
    ax.add_artist(unit_circle)

    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')
    ax.grid(True)

    ax.set_aspect('equal')

    ax.add_artist(
        plt.legend(
            [points, unit_circle], ['Eigenvalues', 'Unit circle'],
            loc=1))
    plt.savefig('../data/dmd/eigenvalues')

def plot_modes(modes, shape, eigenvalues):
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    xgrid, ygrid = np.meshgrid(x, y)
    index_mode = list(range(modes.shape[1]))

    for idx in index_mode:
        fig = plt.figure(figsize=(12, 5))
        title = 'DMD Mode ' + str(idx) + ', eigenvalues = ', str(eigenvalues[idx])
        fig.suptitle(title)

        real_ax = fig.add_subplot(1, 2, 1)
        imag_ax = fig.add_subplot(1, 2, 2)

        mode = modes.T[idx].reshape(xgrid.shape, order='C')

        real = real_ax.pcolor(
            xgrid,
            ygrid,
            mode.real,
            cmap='viridis',
            vmin=mode.real.min(),
            vmax=mode.real.max())
        imag = imag_ax.pcolor(
            xgrid,
            ygrid,
            mode.imag,
            cmap='viridis',
            vmin=mode.imag.min(),
            vmax=mode.imag.max())

        fig.colorbar(real, ax=real_ax)
        fig.colorbar(imag, ax=imag_ax)

        real_ax.set_aspect('auto')
        imag_ax.set_aspect('auto')

        real_ax.invert_yaxis()
        imag_ax.invert_yaxis()

        real_ax.set_title('Real')
        imag_ax.set_title('Imag')

        # padding between elements
        plt.tight_layout(pad=2.)
        filename = '../data/dmd/dmd_mode_' + str(idx)
        plt.savefig(filename)
        plt.close(fig)


