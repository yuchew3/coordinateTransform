import PIL
import pims
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

def load_single_image():
    tiff_file = '../data/SliceImageCa7.tif'
    img = plt.imread(tiff_file)
    img = img - img.min()
    img = img / img.max()
    plt.imshow(img)
    plt.clim(0.05, 1)
    plt.show()

def save_each_frame(img):
    for i in range(4):
        print(img[:,:,i])
        plt.imshow(img[:,:,i])
        plt.show()

def load_video():
    tiff_file = '../data/vid.tif'
    im = io.imread(tiff_file)
#     for i in range(10):
#         plt.imshow(im[i*4000,:,:])
#         plt.show()
    return im

def chop_video():
    tiff_file = '../data/vid.tif'
    im = io.imread(tiff_file)
    print(im.shape)
    short = im[:400, :, :]
    io.imsave('../data/short_vid.tif', short)

    

def convert_video(vid):
    _, _, frames = vid.shape
    print(vid.shape)
#     for i in range(frames):
        

if __name__ == "__main__":
#     vid = load_video()
#     res = convert_video(vid)
    chop_video()