# -*- coding:utf8 -*-
import matplotlib.pyplot as plt
from math import *
from volib import *
import numpy as np
import cv2
import os

plt.ion()

print(cv2.__version__)
path = "dataset_libviso/"

FLANN_INDEX_LSH=6;
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6,
                    key_size = 12, multi_probe_level = 1)
search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params,search_params)
featuredDetector = cv2.BRISK_create()
featuredDescriptor = cv2.BRISK_create()


if __name__ == "__main__":
    oldCorrespondence = None
    imgMN2O = None
    flag = False
    for frameId in range(373):
        imgl = cv2.imread("{0}I1_{1:06d}.png".format(path,frameId))
        imgr = cv2.imread("{0}I2_{1:06d}.png".format(path,frameId))

        #Detect and compute features and descriptors of the current stereo pair
        kptsL = featuredDetector.detect(imgl)
        kptsL, descL = featuredDescriptor.compute(imgl,kptsL)
        kptsR = featuredDetector.detect(imgr)
        kptsR, descR = featuredDescriptor.compute(imgr,kptsR)

        matches = twoWayMatch(flann, kptsL, descL, kptsR, descR)
        matches, kptsL, descL, kptsR, descR, dists = cleanFeatures(matches,
                                                                    kptsL,
                                                                    descL,
                                                                    kptsR,
                                                                    descR)


        #Draw everything
        imgl = cv2.drawKeypoints(imgl,kptsL, None)
        imgr = cv2.drawKeypoints(imgr,kptsR, None)

        drawPyPlotFeatures(imgl, imgr)
        plt.pause(0.05)
        plt.clf()
