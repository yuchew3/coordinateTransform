import matplotlib
matplotlib.use('Agg')
import tkinter as tk
from skimage import io
import cv2
import numpy as np
from PIL import Image
import ca_data_utils

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
        self.sleepCanvas.grid(column=0, row=0)
        self.wake1Canvas.grid(column=0, row=1)
        self.wake2Canvas.grid(column=1, row=1)

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
        self.wake1_covV = np.cov(self.wake1_svmat)
        self.wake2_covV = np.cov(self.wake2_svmat)
        print(self.sleep_covV.shape)
        print('done with cov v')
        self.sleep_varP = np.sum(np.multiply(np.matmul(self.sleep_ur, self.sleep_covV.T).T, self.sleep_ur.T), axis=0)
        print(self.sleep_varP.shape)
        self.wake1_varP = np.sum(np.multiply(np.matmul(self.wake1_ur, self.wake1_covV.T).T, self.wake1_ur.T), axis=0)
        self.wake2_varP = np.sum(np.multiply(np.matmul(self.wake2_ur, self.wake2_covV.T).T, self.wake2_ur.T), axis=0)
        print('done with var p')



        self.showCorr(1, 1)
    
    def showCorr(self, x, y):
        print('show ', x, y)
        ind = 180 * x + y
        sleepCorr = np.matmul(self.sleep_ur[ind], np.matmul(self.sleep_covV, self.sleep_ur.T))
        wake1Corr = np.matmul(self.wake1_ur[ind], np.matmul(self.wake1_covV, self.wake1_ur.T))
        wake2Corr = np.matmul(self.wake2_ur[ind], np.matmul(self.wake2_covV, self.wake2_ur.T))
        print(sleepCorr.shape)
        sleepCorr = sleepCorr / np.sqrt(self.sleep_varP[ind] * self.sleep_varP)
        wake1Corr = wake1Corr / np.sqrt(self.wake1_varP[ind] * self.wake1_varP)
        wake2Corr = wake2Corr / np.sqrt(self.wake2_varP[ind] * self.wake2_varP)


    def get_pixel(self, event, x, y, flags, params):
        self.showCorr(x, y)

def main():
    root = tk.Tk()
    root.title("Correlation Map")
    gui = GUI(root)

    global current_x, current_y
    reference = ca_data_utils.load_sleep_vid()[0]
    cv2.namedWindow("reference", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("reference", gui.get_pixel)
    cv2.imshow('reference', reference)
    window = tk.Tk()

    root.mainloop()

if __name__ == '__main__':
    main()