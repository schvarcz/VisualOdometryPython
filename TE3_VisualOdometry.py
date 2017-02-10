# -*- coding:utf8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import *
import numpy as np
import time
import cv2
import os
plt.ion()

FLANN_INDEX_LSH=6;
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict(checks=100)

print(cv2.__version__)
path = "C:\\Users\\Guilherme\\Desktop\\visualodometry\\dataset_libviso\\"
path = "dataset_libviso/"
pathSalvar = "salvar_1/"
confianca = 10000
distancia_cameras = .57073824147
cameraMatrix = np.asarray([[6.452401e+02,   0           ,   6.601406e+02],
                            [0           ,   6.452401e+02,   2.611004e+02],
                            [0           ,   0           ,   1           ]])
maxDisp = (300,20)

flann = cv2.FlannBasedMatcher(index_params,search_params)
featuredDetector = cv2.BRISK_create()
featuredDescriptor = cv2.BRISK_create()


posCam = np.asarray([.6,.0,1.6])
IMU = [[0],[-.0],[0]]
ODO = [[0.6],[0],[1.6]]

def twoWayMatch(kptsL, descL, kptsR, descR, checkFunc = lambda kptL, kptR, match: True):
    matches_r = []
    matchesL2R = flann.knnMatch(descL,descR,k=2)
    matchesR2L = flann.knnMatch(descR,descL,k=2)
    for mL2R in matchesL2R:
        matchFound = False
        if type(mL2R) == list:
            for iML2R in mL2R:
                if type(matchesR2L[iML2R.trainIdx]) == list:
                    for iMR2L in matchesR2L[iML2R.trainIdx]:
                        if iMR2L.trainIdx == iML2R.queryIdx and checkFunc(kptsL[iML2R.queryIdx], kptsR[iML2R.trainIdx], iML2R):
                            matches_r.append(iML2R)
                            matchFound = True
                            break
                else:
                    if matchesR2L[iML2R.trainIdx] == iML2R.queryIdx and checkFunc(kptsL[iML2R.queryIdx], kptsR[iML2R.trainIdx], iML2R):
                        matches_r.append(iML2R)
                        matchFound = True
                if matchFound :
                    break
        elif mL2R != None:
            if type(matchesR2L[mL2R.trainIdx]) == list:
                for iMR2L in matchesR2L[mL2R.trainIdx]:
                    if iMR2L.trainIdx == mL2R.queryIdx and checkFunc(kptsL[mL2R.queryIdx], kptsR[mL2R.trainIdx], mL2R):
                        matches_r.append(mL2R)
                        matchFound = True
                        break
            else:
                if matchesR2L[mL2R.trainIdx] == mL2R.queryIdx and checkFunc(kptsL[mL2R.queryIdx], kptsR[mL2R.trainIdx], mL2R):
                    matches_r.append(mL2R)
                    matchFound = True
    return matches_r

def cleanFeatures(matches, kptsL, descL, kptsR, descR, coords3D = None):
    matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R, coords3D_R = [], [], [], [], [], [], []
    idx = 0
    for m in matches:
        matches_r.append([cv2.DMatch(idx,idx,m.imgIdx,m.distance)])
        kptsL_R.append(kptsL[m.queryIdx])
        kptsR_R.append(kptsR[m.trainIdx])
        descL_R.append(descL[m.queryIdx])
        descR_R.append(descR[m.trainIdx])
        dist_R.append(m.distance)
        if coords3D != None:
            coords3D_R.append(coords3D[m.trainIdx])
        idx += 1

    descL_R, descR_R = np.asarray(descL_R, np.uint8), np.asarray(descR_R, np.uint8)

    if coords3D != None:
        return matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R, coords3D_R
    return matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R

def cleanInliers(inliers, matches, kptsL, kptsR):
    matches_r, kptsL_R, kptsR_R = [], [], []
    idx = 0
    for i in inliers:
        m = matches[i][0]
        matches_r.append([cv2.DMatch(idx,idx,m.imgIdx,m.distance)])
        kptsL_R.append(kptsL[i])
        kptsR_R.append(kptsR[i])
        idx += 1

    return matches_r, kptsL_R, kptsR_R

def drawFeaturesCorrespondance(imgl,kptsL, kptsR, matches, color = (255,0,0)):
    img = cv2.drawKeypoints(imgl,kptsL, None)
    for m in matches:
        [m] = m
        startPt = int(round(kptsL[m.queryIdx].pt[0])), int(round(kptsL[m.queryIdx].pt[1]))
        endPt   = int(round(kptsR[m.trainIdx].pt[0])), int(round(kptsR[m.trainIdx].pt[1]))
        cv2.arrowedLine(img, startPt, endPt, color)
    return img

def featuresCoordinates(kptsL, descL, kptsR, descR, dists):
    coords = []
    for idx in xrange(len(kptsL)):

        dispX = kptsL[idx].pt[0] - kptsR[idx].pt[0]
        dispY = kptsL[idx].pt[1] - kptsR[idx].pt[1]

        x = (kptsR[idx].pt[0]-cameraMatrix[0,2])*distancia_cameras/dispX
        y = (kptsR[idx].pt[1]-cameraMatrix[1,2])*distancia_cameras/dispX
        z = cameraMatrix[0,0]*distancia_cameras/dispX
        coords.append([x,y,z])

    return coords

def featureCorrespondenceCheck(kptL, kptR, match):
    dispX = kptL.pt[0] - kptR.pt[0]
    dispY = kptL.pt[1] - kptR.pt[1]

    if match.distance > confianca or dispX == 0 or abs(dispX) > maxDisp[0] or abs(dispY) > maxDisp[1]:
        return False
    return True

def kpts2np(kpts):
    return np.asarray([kpt.pt for kpt in kpts])

def drawPyPlot(imgMN2O, IMU, ODO):
    axImg = plt.subplot(211)
    axImg.imshow(imgMN2O)
    axIMU = plt.subplot(223)
    roll, pitch, yaw  = IMU
    axIMU.plot(roll, label="roll")
    axIMU.plot(pitch, label="pitch")
    axIMU.plot(yaw, label="yaw")
    axIMU.set_ylim(-pi,pi)
    axIMU.legend()

    axODO = plt.subplot(224,projection='3d')
    x, y, z  = ODO
    axODO.plot(x, y, z, label="Path")
    axODO.set_xlim3d(-190,0)
    axODO.set_ylim3d(-145,45)
    axODO.set_zlim3d(-60,100)

def drawPyPlotFeatures(imgM, imgMN2O):
    axImg = plt.subplot(211)
    axImg.imshow(imgM)
    axImg = plt.subplot(212)
    axImg.imshow(imgMN2O)

def matrixRot(alpha,beta,gama):
    return np.matrix(
    [ [+cos(beta)*cos(gama), -cos(beta)*sin(gama), +sin(beta)],
    [+sin(alpha)*sin(beta)*cos(gama)+cos(alpha)*sin(gama),-sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama),-sin(alpha)*cos(beta)],
    [-cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama), +cos(alpha)*sin(beta)*sin(gama)+sin(alpha)*cos(gama), +cos(alpha)*cos(beta)]])

if __name__ == "__main__":
    oldCorrespondence = None
    imgMN2O = None
    flag = False
    for i in range(373):
        imgl, imgr = cv2.imread("{0}I1_{1:06d}.png".format(path,i)), cv2.imread("{0}I2_{1:06d}.png".format(path,i))

        #Detect and compute features and descriptors of the current stereo pair
        kptsL = featuredDetector.detect(imgl)
        kptsL, descL = featuredDescriptor.compute(imgl,kptsL)
        kptsR = featuredDetector.detect(imgr)
        kptsR, descR = featuredDescriptor.compute(imgr,kptsR)

        #Compute the two way matching and clean our vectors
        maxDisp = (200,50)
        matches = twoWayMatch(kptsL, descL, kptsR, descR, featureCorrespondenceCheck)
        matches, kptsL, descL, kptsR, descR, dists = cleanFeatures(matches, kptsL, descL, kptsR, descR)

        coords3D = featuresCoordinates(kptsL, descL, kptsR, descR, dists)

        imgM = drawFeaturesCorrespondance(imgl, kptsL, kptsR, matches)

        #Compute egomotion based on the previous stereo pair
        if oldCorrespondence != None:
            matchesOld, kptsLOld, descLOld, kptsROld, descROld, coords3DOld = oldCorrespondence
            maxDisp = (200,100)
            matchesN2O = twoWayMatch(kptsL, descL, kptsLOld, descLOld, featureCorrespondenceCheck)
            matchesN2O, kptsLN, descLN, kptsLOld, descLOld, __, coords3DOld = cleanFeatures(matchesN2O, kptsL, descL, kptsLOld, descLOld, coords3DOld)
            imgMN2O = drawFeaturesCorrespondance(imgl, kptsLN, kptsLOld, matchesN2O)

            coords3DOld = np.asarray(coords3DOld)

            flag, rvec, tvec, inliers = cv2.solvePnPRansac(coords3DOld, kpts2np(kptsLN), cameraMatrix, None)
            # flag, rvec, tvec = cv2.solvePnP(coords3DOld, kpts2np(kptsL), cameraMatrix, None)

            pitch, yaw, roll  = rvec
            rvec = roll, pitch, yaw
            for i in range(3):
                IMU[i].append(IMU[i][-1]+rvec[i])

            roll, pitch, yaw = IMU[0][-1], IMU[1][-1], IMU[2][-1]
            y, z, x  = tvec
            tvec = np.asarray([x,y,z])
            posCam += matrixRot(roll, pitch, yaw).dot(tvec).A1
            for i in range(3):
                ODO[i].append(posCam[i])

            if flag:
                matchesI, kptsLI, kptsLOldI = cleanInliers(inliers, matches, kptsLN, kptsLOld)
                imgMN2O = drawFeaturesCorrespondance(imgMN2O, kptsLI, kptsLOldI, matchesI, (0,0,255))

        if oldCorrespondence == None or flag:
            oldCorrespondence = (matches, kptsL, descL, kptsR, descR, coords3D)

        #Draw everything
        imgl = cv2.drawKeypoints(imgl,kptsL, None)
        imgr = cv2.drawKeypoints(imgr,kptsR, None)

        # cv2.imshow('left', imgl)
        # cv2.imshow('right', imgr)
        # cv2.imshow('matches-stereo', imgM)
        # if imgMN2O != None:
        #     cv2.imshow('matches-nav', imgMN2O)
        # key = cv2.waitKey(33) & 255
        # if key == 27:
        #     break



        if imgMN2O != None:
            drawPyPlot(imgMN2O, IMU, ODO)
            if flag:
                plt.pause(0.05)
            else:
                plt.pause(10)
        plt.clf()
    print(os.getcwd())
plt.ioff()

drawPyPlot(imgMN2O, IMU, ODO)
plt.show()
