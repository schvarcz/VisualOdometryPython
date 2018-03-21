# -*- coding:utf8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import *
import numpy as np
import cv2

def twoWayMatch(flann, kptsL, descL, kptsR, descR, checkFunc = lambda kptL, kptR, match: True):
    """
    Perform two-way matching among two sets of features and their respectives descriptors

    Keyword arguments:

    flann -- FlannBasedMatcher instance to perform the matching procedure.
    kptsL -- Keypoints from the left image or, if you are using images from different steps, the newest image.
    descL -- Respective descriptors from kptsL keypoints.
    kptsR -- Keypoints from the right image or, if you are using images from different steps, the oldest image.
    descR -- Respective descriptors from kptsR keypoints.
    checkFunc -- Final check function to verify a match among two features. Useful to perform any heurist test, like a descriptor distance or euclidean distance thresholds (default: lambda kptL, kptR, match: True)

    Returns list of matches from left to right.
    """
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

def twoWayMatchAndLoweTest(flann, kptsL, descL, kptsR, descR, checkFunc = lambda kptL, kptR, match: True):
    """
    Perform two-way matching among two sets of features and their respectives descriptors

    Keyword arguments:

    flann -- FlannBasedMatcher instance to perform the matching procedure.
    kptsL -- Keypoints from the left image or, if you are using images from different steps, the newest image.
    descL -- Respective descriptors from kptsL keypoints.
    kptsR -- Keypoints from the right image or, if you are using images from different steps, the oldest image.
    descR -- Respective descriptors from kptsR keypoints.
    checkFunc -- Final check function to verify a match among two features. Useful to perform any heurist test, like a descriptor distance or euclidean distance thresholds (default: lambda kptL, kptR, match: True)

    Returns list of matches from left to right.
    """
    matches_r = []
    matchesL2R = flann.knnMatch(descL,descR,k=2)
    matchesR2L = flann.knnMatch(descR,descL,k=2)
    for mL2R in matchesL2R:
        matchFound = False
        if type(mL2R) == list:
            for i in range(len(mL2R)):
                iML2R = mL2R[i]
                if type(matchesR2L[iML2R.trainIdx]) == list:
                    for iMR2L in matchesR2L[iML2R.trainIdx]:
                        if iMR2L.trainIdx == iML2R.queryIdx and checkFunc(kptsL[iML2R.queryIdx], kptsR[iML2R.trainIdx], iML2R):
                            if i < len(mL2R)-1:
                                m, n = iML2R, mL2R[i+1]
                                if m.distance < 0.8*n.distance:
                                    matches_r.append(iML2R)
                                    matchFound = True
                                    break
                            else:
                                matches_r.append(iML2R)
                                matchFound = True
                                break
                else:
                    if matchesR2L[iML2R.trainIdx] == iML2R.queryIdx and checkFunc(kptsL[iML2R.queryIdx], kptsR[iML2R.trainIdx], iML2R):
                        if i < len(mL2R)-1:
                            m, n = iML2R, mL2R[i+1]
                            if m.distance < 0.8*n.distance:
                                matches_r.append(iML2R)
                                matchFound = True
                                break
                        else:
                            matches_r.append(iML2R)
                            matchFound = True
                            break
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

def loweMatchingTest(flann, kptsL, descL, kptsR, descR, checkFunc = lambda kptL, kptR, match: True):
    matches_r = []
    matchesL2R = flann.knnMatch(descL,descR,k=2)
    for mL2R in matchesL2R:
        if len(mL2R) == 2:
            m, n = mL2R
            if m.distance < 0.8*n.distance and checkFunc(kptsL[mL2R.queryIdx], kptsR[mL2R.trainIdx], mL2R):
                matches_r.append(m)
        else:
            matches_r.append(mL2R[0])
    return matches_r

def cleanFeatures(matches, kptsL, descL, kptsR, descR, coords3D = None):
    """
    Clean kptsL, descL, kptsR, descR and coords3D vectors according to the informed matches.

    Returns A list for each informed parameter but matches. A list for coords3D is only returned if a coords3D is informed.
    """
    matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R, coords3D_R = [], [], [], [], [], [], []
    idx = 0
    for m in matches:
        matches_r.append([cv2.DMatch(idx,idx,m.imgIdx,m.distance)])
        kptsL_R.append(kptsL[m.queryIdx])
        kptsR_R.append(kptsR[m.trainIdx])
        descL_R.append(descL[m.queryIdx])
        descR_R.append(descR[m.trainIdx])
        dist_R.append(m.distance)
        if not coords3D is None:
            coords3D_R.append(coords3D[m.trainIdx])
        idx += 1

    descL_R, descR_R = np.asarray(descL_R, np.uint8), np.asarray(descR_R, np.uint8)

    if not coords3D is None:
        return matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R, np.asarray(coords3D_R)
    return matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R

def cleanInliers(inliers, matches, kptsL, kptsR):
    """
    Clean matches, kptsL and kptsR according to the informed inlier indexes.

    Returns A list for each infromed parameter, except inliers, containing only the respective inliers information.
    """
    matches_r, kptsL_R, kptsR_R = [], [], []
    idx = 0
    for [i] in inliers:
        m = matches[i][0]
        matches_r.append([cv2.DMatch(idx,idx,m.imgIdx,m.distance)])
        kptsL_R.append(kptsL[i])
        kptsR_R.append(kptsR[i])
        idx += 1
    return matches_r, kptsL_R, kptsR_R

def cleanInliersFundamentalMat(inliers, matches, kptsL, kptsR):
    """
    Clean matches, kptsL and kptsR according to the informed inlier indexes.

    Returns A list for each infromed parameter, except inliers, containing only the respective inliers information.
    """
    matches_r, kptsL_R, kptsR_R = [], [], []
    idx = 0
    for i in range(len(inliers)):
        if inliers[i][0] == 1:
            m = matches[i][0]
            matches_r.append([cv2.DMatch(idx,idx,m.imgIdx,m.distance)])
            kptsL_R.append(kptsL[i])
            kptsR_R.append(kptsR[i])
            idx += 1
    return matches_r, kptsL_R, kptsR_R

def kpts2np(kpts):
    """
    Converts keypoints to a numpy matrix to be used with cv2.solvePnP and cv2.solvePnPRansac
    """
    return np.asarray([[kpt.pt] for kpt in kpts])

def drawFeaturesCorrespondance(imgl, kptsL, kptsR, matches, color = (255,0,0)):
    img = cv2.drawKeypoints(imgl,kptsL, None)
    for m in matches:
        [m] = m
        startPt = int(round(kptsL[m.queryIdx].pt[0])), int(round(kptsL[m.queryIdx].pt[1]))
        endPt   = int(round(kptsR[m.trainIdx].pt[0])), int(round(kptsR[m.trainIdx].pt[1]))
        cv2.arrowedLine(img, startPt, endPt, color)
    return img

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
    axODO.set_xlim3d(-170,20)
    axODO.set_ylim3d(-15,175)
    axODO.set_zlim3d(-50,100)

def drawPyPlotFeatures(imgM, imgMN2O):
    axImg = plt.subplot(211)
    axImg.imshow(imgM)
    axImg = plt.subplot(212)
    axImg.imshow(imgMN2O)

def matrixRot(alpha, beta, gama):
    """
    Computes a rotation matrix in 3D space

    Keyword arguments:

    alpha -- Corresponde to roll angle. Rotation around X axis.
    beta -- Corresponde to pitch angle. Rotation around Y axis.
    gama -- Corresponde to Yaw angle. Rotation around Z axis.

    Returns A numpy matrix encoding a 3D rotation.
    """
    return np.matrix(
    [ [+cos(beta)*cos(gama), -cos(beta)*sin(gama), +sin(beta)],
    [+sin(alpha)*sin(beta)*cos(gama)+cos(alpha)*sin(gama),-sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama),-sin(alpha)*cos(beta)],
    [-cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama), +cos(alpha)*sin(beta)*sin(gama)+sin(alpha)*cos(gama), +cos(alpha)*cos(beta)]])
