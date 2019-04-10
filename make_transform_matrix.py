import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
import numpy as np
import cv2
from skimage import io

def main():
    root = Tk()
    root.withdraw()
    vid_dir = 'short_vid.tif'
    first_frame = load_first_frame(vid_dir)
    template_dir = '../data/cortical_map_top_down.png'
    template = cv2.imread(template_dir)
    r, c,_ = template.shape
    template = cv2.resize(template, (1140,1320))
    while True:
        # open picture to get points
        from_points, to_points = get_match_points(first_frame, template)
        # make transform matrix
        M = transform(from_points, to_points, first_frame, template)
        MsgBox = messagebox.askquestion('Done with mapping','Are you sure you want to save this transform matrix?')
        print(MsgBox)
        if MsgBox == 'yes':
            save_dir = filedialog.asksaveasfilename(initialdir='~/Desktop/Senior/Research/data')
            np.save(save_dir, M)
            break
        plt.close()


def transform(from_points, to_points, first_frame, template):
    M = cv2.getPerspectiveTransform(from_points, to_points)
    r, c,_ = template.shape
    dst = cv2.warpPerspective(first_frame,M,(c,r), borderValue=np.nan)
    plt.axis('off')
    plt.imshow(template, alpha=0.5)
    plt.imshow(dst, alpha=0.9)
    plt.show(block=False)
    return M

def get_match_points(first_frame, template):
    global pixels_1, pixels_2
    pixels_1 = list()
    pixels_2 = list()
    get_pixels(first_frame, template)
    from_points = np.float32(pixels_1)
    print(from_points)
    to_points = np.float32(pixels_2)
    print(to_points)
    return from_points, to_points

def load_first_frame(vid_dir):
    im = io.imread(vid_dir)
    return im[0]

def mouse_callback(event, x, y, flags, params):
    # click event
    if event == 1:
        global pixels, showing_img
        cv2.circle(showing_img, (x,y), 5, (255,0,0), -1)
        pixels.append([x, y])

def mouse_callback_1(event, x, y, flags, params):
    global pixels_1, showing_img_1, original_1
    if event == 1:
        cv2.circle(showing_img_1, (x,y), 5, (255,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(showing_img_1, str(len(pixels_1)+1),(x,y),font,1,(255,0,0),2,cv2.LINE_AA)
        pixels_1.append([x,y])
    elif event == cv2.EVENT_FLAG_RBUTTON:
        showing_img_1 = original_1.copy()
        pixels_1 = pixels_1[:-1]
        i = 1
        for (a, b) in pixels_1:
            cv2.circle(showing_img_1, (a,b), 5, (255,0,0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(showing_img_1, str(i),(a,b),font,1,(255,0,0),2,cv2.LINE_AA)
            i += 1

def mouse_callback_2(event, x, y, flags, params):
    if event == 1:
        global pixels_2, showing_img_2
        cv2.circle(showing_img_2, (x,y), 5, (255,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(showing_img_2, str(len(pixels_2)+1),(x,y),font,2,(255,0,0),2,cv2.LINE_AA)
        pixels_2.append([x,y])

def get_pixels(img, template):
    cv2.namedWindow("image1", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image1", mouse_callback_1)
    cv2.namedWindow('image2', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image2", mouse_callback_2)
    global showing_img_1, showing_img_2, original_1, original_2
    original_1 = img
    original_2 = template
    showing_img_1 = img.copy()
    showing_img_2 = template.copy()
    while True:
        cv2.imshow('image1', showing_img_1)
        cv2.imshow('image2', showing_img_2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow('image1')
    cv2.destroyWindow('image2')


def get_pixels_old(img):
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", mouse_callback)
    global showing_img
    showing_img = img.copy()
    while True:
        cv2.imshow("image", showing_img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow("image")

if __name__ == '__main__':
    main()

