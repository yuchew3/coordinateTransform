import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def tryit():
    img = ("../data/sliceImage.png")
    rows, cols, ch = img.shape
    print(rows)
    print(cols)

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

    pts1 = np.float32([[56,65],[868,52],[28,887],[889,890]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(300,300))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def save_img():
    img = cv2.imread("../data/sliceImage.png")
    rows, cols, ch = img.shape

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("compare.png", dst)

def mouse_callback(event, x, y, flags, params):
    # click event
    if event == 1:
        global pixels
        #store the coordinates of the click event
        pixels.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(pixels)

def get_pixels(first):
    if (first):
        arr = np.load("slice.npy")
        print(arr)
        ma = arr.max()
        mi = arr.min()
        img = (arr - mi) / (ma - mi)
        print(img)
        
    else:
        img = np.load("../data/flat_cortex_template.npy")
    print(img.shape)
    # scale_width = 640 / img.shape[1]
    # scale_height = 480 / img.shape[0]
    # scale = min(scale_width, scale_height)
    # window_width = int(img.shape[1] * scale)
    # window_height = int(img.shape[0] * scale)
    
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow("image", window_width, window_height)
    cv2.setMouseCallback("image", mouse_callback)
    cv2.imshow("image", img)
    cv2.waitKey(0)

def transform():
    global pixels
    get_pixels(True)
    from_pixels = np.float32(pixels)
    pixels = list()
    get_pixels(False)
    to_pixels = np.float32(pixels)

    print(from_pixels)
    print(to_pixels)

    img = np.load("slice.npy")
    img = (img - img.min()) / (img.max() - img.min())
    M = cv2.getPerspectiveTransform(from_pixels, to_pixels)

    np.save("../data/transformMatrix", M)

    compare = np.load("../data/flat_cortex_template.npy")
    r, c = compare.shape
    dst = cv2.warpPerspective(img,M,(c,r))  
    # tmp = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    # b, g, r = cv2.split(dst)
    # rgba = [b,g,r, alpha]
    # dst = cv2.merge(rgba,4)

    plt.imshow(compare, alpha=0.5)
    plt.imshow(dst, alpha=0.9)

    plt.axis('off')
   
    # plt.subplot(221),plt.imshow(img),plt.title('Input')
    # plt.subplot(222),plt.imshow(dst),plt.title('Output')
    # plt.subplot(224),plt.imshow(compare),plt.title('Reference')
    plt.show()

def try_tps_transformer():
    tps=cv2.createThinPlateSplineShapeTransformer()



if __name__ == "__main__":
    pixels = list()
    transform()