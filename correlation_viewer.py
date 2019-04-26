import matplotlib
matplotlib.use('Agg')
import tkinter as tk
from skimage import io
import cv2
import numpy as np
from PIL import Image

class GUI:
    def __init__(self, master):
        self.master = master
        self.sleepCanvas = tk.Frame(self.master)
        self.wake1Canvas = tk.Frame(self.master)
        self.wake2Canvas = tk.Frame(self.master)
        for r in range(2):
            self.master.rowconfigure(r, weight=1)    
        for c in range(2):
            self.master.columnconfigure(c, weight=1)
        self.sleepCanvas.grid(column=0, row=0, sticky = W+E+N+S)
        self.wake1Canvas.grid(column=0, row=1, sticky = W+E+N+S)
        self.wake2Canvas.grid(column=1, row=1, sticky = W+E+N+S)

        self.sleep_umat = np.load('../data/byState/sleep_umat.npy')
        self.sleep_svmat = np.load('../data/byState/sleep_svmat.npy')
        self.wake1_umat = np.load('../data/byState/wake1_umat.npy')
        self.wake1_svmat = np.load('../data/byState/wake1_svmat.npy')
        self.wake2_umat = np.load('../data/byState/wake2_umat.npy')
        self.wake2_svmat = np.load('../data/byState/wake2_svmat.npy')

        print('pre-computing...')

        self.sleep_ur = self.sleep_umat.reshape((43200, 1000))
        self.wake1_ur = self.wake1_umat.reshape((43200, 1000))
        self.wake2_ur = self.wake2_umat.reshape((43200, 1000))
        print('done with u matrix reshape')
        self.sleep_covV = np.cov(self.sleep_svmat)
        print(slef.sleep_covV.shape)
        self.wake1_covV = np.cov(self.wake1_svmat)
        self.wake2_covV = np.cov(self.wake2_svmat)

        print('done with cov v')
        self.sleep_varP = np.dot(np.matmul(self.sleep_ur, self.sleep_covV), self.sleep_ur)
        print(self.sleep_varP.shape)
        self.wake1_varP = np.dot(np.matmul(self.wake1_ur, self.wake1_covV), self.wake1_ur)
        self.wake2_varP = np.dot(np.matmul(self.wake2_ur, self.wake2_covV), self.wake2_ur)
        print('done with var p')



        self.showCorr(1, 1)
    
    def showCorr(self, x, y):
        print('show ', x, y)

    def get_pixel(self, event, x, y, flags, params):
        self.showCorr(x, y)

def main():
    root = tk.Tk()
    root.title("Correlation Map")
    gui = GUI(root)

    global current_x, current_y
    reference = cv2.imread('../data/SliceImageCa7.png')
    cv2.namedWindow("reference", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("reference", gui.get_pixel)
    cv2.imshow('reference', reference)
    window = tkinter.Tk()

    root.mainloop()

if __name__ == '__main__':
    main()