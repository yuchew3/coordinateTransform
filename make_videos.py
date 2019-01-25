import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pickle
from scipy.sparse import csr_matrix

def overlay1():
    matrix = np.load("matrix.npy")
    compare = np.load("../data/flat_cortex_template.npy")
    original_video = io.imread('short_vid.tif')

    first_frame = original_video[1,:,:]

    limit = 4.18187
    norm = mpl.colors.Normalize(vmin = -limit, vmax = limit)
    frames, _, _ = original_video.shape

    top_down_overlay = plt.imread("../data/cortical_map_top_down.png")
    flat_view = compare
    pixel = cv2.warpPerspective(first_frame,matrix,(1320,1140))
    # plot & params
    fig, ax = plt.subplots(figsize=(6, 6))
    # plot source voxel
    im = plt.pcolormesh(pixel, zorder=1, cmap='bwr', norm=norm)
    plt.gca().invert_yaxis() # flips yaxis
    plt.axis('off')
    # plot overlay THIS IS THEY KEY PART... zorder sets what's on top
    extent = plt.gca().get_xlim() + plt.gca().get_ylim()
    print(extent)
    print(top_down_overlay.shape)
    plt.imshow(top_down_overlay, interpolation="nearest", extent=extent, zorder=2)
    # add colorbar
    cbar = plt.colorbar(im, shrink=0.3, use_gridspec=True)
    cbar.ax.tick_params(labelsize=6) 
    plt.tight_layout()
    plt.savefig('try3',
                bbox_inches=None, facecolor=None, edgecolor=None,
                transparent=True, dpi=240)
    plt.close()


if __name__ == '__main__':
    matrix = np.load("../data/matrix.npy")
    compare = np.load("../data/flat_cortex_template.npy")
    original_video = io.imread('../data/vid.tif')

    first_frame = original_video[1,:,:]

    maxi = 4.18187
    mini = -0.2908125
    norm = mpl.colors.Normalize(vmin = mini, vmax = maxi)
    frames, _, _ = original_video.shape

    top_down_overlay = plt.imread("../data/cortical_map_top_down.png")

    for i in range(frames):
        fig, ax = plt.subplots(figsize=(16,12))
        img = cv2.warpPerspective(original_video[i,:,:],matrix,(1320,1140))
        cax = ax.imshow(img, cmap='inferno', norm=norm)
        ax.imshow(top_down_overlay, extent=(0, 1140, 1320, 0))
        
        cbar = fig.colorbar(cax, ticks=[], orientation='vertical', norm=norm, shrink=0.5)
        plt.axis('off')
        fname = '../overlay_vid/image_{0:05d}'.format(i)
        print(fname)
        fig.savefig(fname)
        plt.close(fig)