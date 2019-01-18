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
    tiff_file = 'short_vid.tif'
    im = io.imread(tiff_file)
    return im

def to_mp4():
    tiff_file = 'short_vid.tif'
    vid = io.imread(tiff_file)
    img = (vid - vid.min()) / (vid.max() - vid.min()) * 255
    img = np.uint8(img)
    frames, r, c = img.shape
    writer = cv2.VideoWriter('short_vid.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (c, r), isColor=False)
    for i in range(frames):
        f = img[i,:,:]
        writer.write(f)
    writer.release()
    

def convert_video(vid):
    matrix = np.load("matrix.npy")
    compare = np.load("../data/flat_cortex_template.npy")
    frames, r, c = vid.shape
    new_vid = np.zeros((frames, 1000, 1000))
    print(vid.shape)
    for i in range(frames):
        new_vid[i, :, :] = cv2.warpPerspective(vid[i,:,:],matrix,(1000,1000))
        # cv2.imshow("image",new_vid[i,:,:])
        # cv2.waitKey(0)
        # return
    io.imsave("../data/converted_vid1.tif", new_vid)
    print("done!")

def convert_large_video():
    vid = io.imread('../data/vid.tif')
    matrix = np.load("../data/matrix.npy")
    compare = np.load("../data/flat_cortex_template.npy")
    frames, r, c = vid.shape
    new_vid = np.memmap('use_large_convert.npy', dtype=np.float32, mode='w+', shape=(frames, 1000, 1000))
    for i in range(frames):
        new_vid[i,:,:] = cv2.warpPerspective(vid[i,:,:],matrix,(1000,1000))
    ma = np.amax(new_vid)
    mi = np.amin(new_vid)
    writer = cv2.VideoWriter('convert_vid.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (c, r), isColor=False)
    for i in range(frames):
        f = np.uint8(255*(new_vid[i,:,:]-mi)/(ma-mi))
        writer.write(f)
    writer.release()
    

def background_extraction():
    tiff_file = '../data/converted_vid1.tif'
    vid = io.imread(tiff_file)
    img = np.average(vid, axis=0)
    print(img.shape)
    frames, r, c = vid.shape
    new_vid = np.zeros(vid.shape)
    for i in range(frames):
        new_vid[i,:,:] = abs(vid[i,:,:] - img)

    io.imsave("../data/extract_vid_abs.tif", new_vid)
    print("done extracting background")

        
        

if __name__ == "__main__":
    #  vid = load_video()
    #  convert_video(vid)
    # background_extraction()
    # to_mp4()
    convert_large_video()
