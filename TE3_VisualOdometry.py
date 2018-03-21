# -*- coding:utf8 -*-
import matplotlib.pyplot as plt
from math import *
from volib import *
import numpy as np
import cv2
import os

plt.ion()

# Python 2.4
FLANN_INDEX_LSH=6;
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6,
                    key_size = 12, multi_probe_level = 1)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Python 3.1
flann = cv2.BFMatcher()

# Python 3.2
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params,search_params)


print(cv2.__version__)
# path = "C:\\Users\\Guilherme\\Dropbox\\Doutorado ENSTA\\visualodometry\\dataset_libviso\\"
# pathSalvar = "C:\\Users\\Guilherme\\Dropbox\\Doutorado ENSTA\\visualodometry\\results\\brisk_orb\\"
path = "./dataset_libviso/"
pathSalvar = "./results/trash/"

if not os.path.exists(pathSalvar):
    os.mkdir(pathSalvar)
logResults = open(pathSalvar+"logfile.csv","w")

confianca = 10000
distancia_cameras = .57073824147
cameraMatrix = np.asarray([[6.452401e+02,   0           ,   6.601406e+02],
                            [0           ,   6.452401e+02,   2.611004e+02],
                            [0           ,   0           ,   1           ]])

maxDispStereo = (100,20)
maxDispTensor = (200,100)
maxDisp = maxDispStereo


# Python 2.4
# featuredDetector = cv2.Feature2D_create("BRISK")
# featuredDescriptor = cv2.Feature2D_create("BRISK")

# Python 3.X
featuredDetector = cv2.BRISK_create()
featuredDescriptor = cv2.BRISK_create()

posCam = np.asarray([.6,.0,1.6])
IMU = [[0],[-.08],[0]]
ODO = [[0.6],[0],[1.6]]

# Students should implement
def featuresCoordinates(kptsL, descL, kptsR, descR, dists):
    coords = []
    for idx in range(len(kptsL)):

        dispX = kptsL[idx].pt[0] - kptsR[idx].pt[0]
        dispY = kptsL[idx].pt[1] - kptsR[idx].pt[1]

        x = (kptsR[idx].pt[0]-cameraMatrix[0,2])*distancia_cameras/dispX
        y = (kptsR[idx].pt[1]-cameraMatrix[1,2])*distancia_cameras/dispX
        z = cameraMatrix[0,0]*distancia_cameras/dispX
        coords.append([[x,y,z]])
    return np.asarray(coords)

def featureCorrespondenceCheck(kptL, kptR, match):
    dispX = kptL.pt[0] - kptR.pt[0]
    dispY = kptL.pt[1] - kptR.pt[1]
    if match.distance > confianca or dispX == 0 or abs(dispX) > maxDisp[0] or abs(dispY) > maxDisp[1]:
        return False
    return True

if __name__ == "__main__":
    oldCorrespondence = None
    imgMN2O = None
    flag = False
    for frameId in range(373):
        imgl, imgr = cv2.imread("{0}I1_{1:06d}.png".format(path,frameId)), cv2.imread("{0}I2_{1:06d}.png".format(path,frameId))

        #Detect and compute features and descriptors of the current stereo pair
        kptsL = featuredDetector.detect(imgl)
        kptsL, descL = featuredDescriptor.compute(imgl,kptsL)
        kptsR = featuredDetector.detect(imgr)
        kptsR, descR = featuredDescriptor.compute(imgr,kptsR)

        #Compute the two way matching and clean our vectors
        maxDisp = maxDispStereo
        matches = twoWayMatch(flann, kptsL, descL, kptsR, descR, featureCorrespondenceCheck)
        matches, kptsL, descL, kptsR, descR, dists = cleanFeatures(matches, kptsL, descL, kptsR, descR)

        coords3D = featuresCoordinates(kptsL, descL, kptsR, descR, dists)

        imgM = drawFeaturesCorrespondance(imgl, kptsL, kptsR, matches)

        #Compute egomotion based on the previous stereo pair
        if oldCorrespondence != None:
            matchesOld, kptsLOld, descLOld, kptsROld, descROld, coords3DOld = oldCorrespondence
            maxDisp = maxDispTensor
            matchesN2O = twoWayMatch(flann, kptsL, descL, kptsLOld, descLOld, featureCorrespondenceCheck)
            matchesN2O, kptsLN, descLN, kptsLOld, descLOld, __, coords3DOld = cleanFeatures(matchesN2O, kptsL, descL, kptsLOld, descLOld, coords3DOld)
            imgMN2O = drawFeaturesCorrespondance(imgl, kptsLN, kptsLOld, matchesN2O)

            kptsLN_np = kpts2np(kptsLN)

            # Python 2.4
            # rvec, tvec, inliers = cv2.solvePnPRansac(coords3DOld, kptsLN_np, cameraMatrix, None)
            # flag = False
            # if len(inliers) > 0:
            #     flag = True

            # Python 3.x
            flag, rvec, tvec, inliers = cv2.solvePnPRansac(coords3DOld, kptsLN_np, cameraMatrix, None)
            # flag, rvec, tvec = cv2.solvePnP(coords3DOld, kptsLN_np, cameraMatrix, None)

            if flag:
                pointsProjected, _ = cv2.projectPoints(coords3DOld, rvec, tvec, cameraMatrix, None)
                error = cv2.norm(kptsLN_np, pointsProjected, cv2.NORM_L2)/len(pointsProjected)

                print(rvec)
                print(tvec)
                print ""
                [pitch], [yaw], [roll]  = rvec
                rvec = roll, pitch, yaw
                for i in range(3):
                    IMU[i].append(IMU[i][-1]+rvec[i])

                roll, pitch, yaw = IMU[0][-1], IMU[1][-1], IMU[2][-1]
                y, z, x  = tvec
                tvec = np.asarray([x,y,z])
                posCam += matrixRot(roll, pitch, yaw).dot(tvec).A1

                x, y, z  = posCam
                for i in range(3):
                    ODO[i].append(posCam[i])

                matchesI, kptsLI, kptsLOldI = cleanInliers(inliers, matches, kptsLN, kptsLOld)
                imgMN2O = drawFeaturesCorrespondance(imgMN2O, kptsLI, kptsLOldI, matchesI, (0,0,255))
                log = ",".join(["{0:06d}".format(frameId), str(x), str(y), str(z), str(roll), str(pitch), str(yaw), str(len(coords3DOld)), str(len(inliers)), str(error)])
                logResults.write(log+"\n")

        if oldCorrespondence == None or flag:
            oldCorrespondence = (matches, kptsL, descL, kptsR, descR, coords3D)

        #Draw everything
        imgl = cv2.drawKeypoints(imgl,kptsL, None)
        imgr = cv2.drawKeypoints(imgr,kptsR, None)

        if not imgMN2O is None:
            #drawPyPlotFeatures(imgM, imgMN2O)
            drawPyPlot(imgMN2O, IMU, ODO)
            cv2.imwrite("{0}I_{1:06d}.png".format(pathSalvar,frameId), imgMN2O)
            if flag:
                plt.pause(0.05)
            else:
                plt.pause(10)
        plt.clf()
    print(os.getcwd())
    plt.ioff()

    logResults.close()
    drawPyPlot(imgMN2O, IMU, ODO)
    plt.show()
