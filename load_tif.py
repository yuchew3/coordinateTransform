import cv2
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
    return im

def to_mp4(vid):
#    tiff_file = 'converted_short_vid.tif'
#    img = io.imread(tiff_file)
    img = (vid - vid.min()) / (vid.max() - vid.min()) * 255
    img = np.uint8(img)
#     img = cv2.cvtColor(img, cv2.GRAY2RGB)
    frames, r, c = img.shape
    writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (c, r), isColor=False)
    for i in range(frames):
        f = img[i,:,:]
        writer.write(f)
    writer.release()
    

def convert_video(vid):
    matrix = np.load("matrix.npy")
    compare = np.load("../data/flat_cortex_template.npy")
    frames, r, c = vid.shape
    new_vid = np.zeros((frames, r, c))
    print(vid.shape)
    for i in range(frames):
        new_vid[i, :, :] = cv2.warpPerspective(vid[i,:,:],matrix,(c,r))
    io.imsave("../data/converted_vid.tif", new_vid)
    print("done!")
        
        

if __name__ == "__main__":
     vid = load_video()
     convert_video(vid)
     to_mp4(vid)
