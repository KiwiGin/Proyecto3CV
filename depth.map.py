import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

class DepthMap:
    def __init__(self, showImages):
        root = os.getcwd()
        imgLeftPath = os.path.join(root, 'demoImages//cubo//TESTLEFT.jpg')
        imgRightPath = os.path.join(root, 'demoImages//cubo//TESTRIGHT.jpg')
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)

        # Verificar si las imágenes se cargaron correctamente
        if self.imgLeft is None or self.imgRight is None:
            print("Error: No se pudo cargar una o ambas imágenes.")
            return

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft, cmap='gray')
            plt.subplot(122)
            plt.imshow(self.imgRight, cmap='gray')
            plt.show()

    def computeDepthMapBM(self):
        nDispFactor = 10  # Número de disparidades a considerar
        stereo = cv.StereoBM.create(numDisparities=16 * nDispFactor, blockSize=21)
        disparity = stereo.compute(self.imgLeft, self.imgRight)
        plt.imshow(disparity, cmap='gray')
        plt.show()

    def computeDepthMapSGBM(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 14  # Número de disparidades a considerar
        num_disp = 16 * nDispFactor - min_disp
        num_disp = (num_disp // 16) * 16  # Asegurarse de que sea múltiplo de 16

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Calcular el mapa de disparidad
        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 16.0

        plt.imshow(disparity, cmap='gray')
        plt.colorbar()
        plt.show()

def demoViewPics():
    # Ver las imágenes
    dp = DepthMap(showImages=True)

def demoStereoBM():
    # Mapa de profundidad usando Block Matching
    dp = DepthMap(showImages=False)
    dp.computeDepthMapBM()

def demoStereoSGBM():
    # Mapa de profundidad usando Semi-Global Block Matching
    dp = DepthMap(showImages=False)
    dp.computeDepthMapSGBM()

if __name__ == "__main__":
    #descomentar la funcion a usar
    # demoStereoSGBM()
     demoStereoBM()
    # demoViewPics()
