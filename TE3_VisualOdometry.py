import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

FLANN_INDEX_LSH=6;
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict(checks=100)

print(cv2.__version__)
path = "C:\\Users\\Guilherme\\Desktop\\visualodometry\\dataset_libviso\\"
foco = 6.452401e+02
distancia_cameras = .57073824147
confianca = 10000
pathSalvar = "salvar_1/"
centroProjecao = (6.601406e+02,2.611004e+02)
maxDisp = (50,20)

flann = cv2.FlannBasedMatcher(index_params,search_params)
featuredDetector = cv2.AKAZE_create()
featuredDescriptor = cv2.AKAZE_create()


def cleanFeatures(matches, kptsL, descL, kptsR, descR):
    matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R = [], [], [], [], [], []
    idx = 0
    for m in matches:
        matches_r.append([cv2.DMatch(idx,idx,m.imgIdx,m.distance)])
        kptsL_R.append(kptsL[m.queryIdx])
        kptsR_R.append(kptsR[m.trainIdx])
        descL_R.append(descL[m.queryIdx])
        descR_R.append(descR[m.trainIdx])
        dist_R.append(m.distance)
        idx += 1
    return matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R

def featureCorrespondenceCheck(kptL, kptR, match):
    dispX = kptL.pt[0] - kptR.pt[0]
    dispY = kptL.pt[1] - kptR.pt[1]

    if match.distance > confianca or dispX == 0 or abs(dispX) > maxDisp[0] or abs(dispY) > maxDisp[1]:
        return False
    return True

def twoWayMatch(kptsL, descL, kptsR, descR):
    matches_r = []
    matchesL2R = flann.knnMatch(descL,descR,k=2)
    matchesR2L = flann.knnMatch(descR,descL,k=2)
    for mL2R in matchesL2R:
        matchFound = False
        if type(mL2R) == list:
            for iML2R in mL2R:
                if type(matchesR2L[iML2R.trainIdx]) == list:
                    for iMR2L in matchesR2L[iML2R.trainIdx]:
                        if iMR2L.trainIdx == iML2R.queryIdx and featureCorrespondenceCheck(kptsL[iML2R.queryIdx], kptsR[iML2R.trainIdx], iML2R):
                            matches_r.append(iML2R)
                            matchFound = True
                            break
                else:
                    if matchesR2L[iML2R.trainIdx] == iML2R.queryIdx and featureCorrespondenceCheck(kptsL[iML2R.queryIdx], kptsR[iML2R.trainIdx], iML2R):
                        matches_r.append(iML2R)
                        matchFound = True
                if matchFound :
                    break
        elif mL2R != None:
            if type(matchesR2L[mL2R.trainIdx]) == list:
                for iMR2L in matchesR2L[mL2R.trainIdx]:
                    if iMR2L.trainIdx == mL2R.queryIdx and featureCorrespondenceCheck(kptsL[mL2R.queryIdx], kptsR[mL2R.trainIdx], mL2R):
                        matches_r.append(mL2R)
                        matchFound = True
                        break
            else:
                if matchesR2L[mL2R.trainIdx] == mL2R.queryIdx and featureCorrespondenceCheck(kptsL[mL2R.queryIdx], kptsR[mL2R.trainIdx], mL2R):
                    matches_r.append(mL2R)
                    matchFound = True
    return matches_r

def featuresCoordinates(kptsL, descL, kptsR, descR, dists):
    coords, kptsL_R, ktpsR_R = [], [], []
    for idx in xrange(len(kptsL)):

        dispX = kptsL[idx].pt[0] - kptsR[idx].pt[0]
        dispY = kptsL[idx].pt[1] - kptsR[idx].pt[1]

        x = (kptsR[idx].pt[0]-centroProjecao[0])*distancia_cameras/dispX
        y = (kptsR[idx].pt[1]-centroProjecao[1])*distancia_cameras/dispX
        z = foco*distancia_cameras/dispX
        coords.append([x,y,z,1.])

    return coords, kptsL_R, kptsR_R

def drawFeaturesCorrespondance(imgl,kptsL, kptsR, matches):
    img = cv2.drawKeypoints(imgl,kptsL, None)
    for m in matches:
        [m] = m
        startPt = int(round(kptsL[m.queryIdx].pt[0])), int(round(kptsL[m.queryIdx].pt[1]))
        endPt   = int(round(kptsR[m.trainIdx].pt[0])), int(round(kptsR[m.trainIdx].pt[1]))
        cv2.arrowedLine(img, startPt, endPt, (255,0,0))
    return img

if __name__ == "__main__":
    oldCorrespondence = None
    imgMN2O = None
    for i in range(373):
        imgl, imgr = cv2.imread("{0}I1_{1:06d}.png".format(path,i)), cv2.imread("{0}I2_{1:06d}.png".format(path,i))

        kptsL = featuredDetector.detect(imgl)
        kptsL, descL = featuredDescriptor.compute(imgl,kptsL)
        kptsR = featuredDetector.detect(imgr)
        kptsR, descR = featuredDescriptor.compute(imgr,kptsR)

        matches = twoWayMatch(kptsL, descL, kptsR, descR)

        matches, kptsL, descL, kptsR, descR, dists = cleanFeatures(matches, kptsL, descL, kptsR, descR)

        # if(oldCorrespondence != None):
        #     matchesOld, kptsLOld, descLOld, kptsROld, descROld = oldCorrespondence
        #
        #     print descR[0]
        #     print descROld[0]
        #     exit()
        #     matchesN2O = twoWayMatch(kptsR, descR, kptsROld, descROld)
        #
        #     matchesN2O, kptsR, descR, kptsROld, descROld, __ = cleanFeatures(matchesN2O, kptsR, descR, kptsROld, descROld)
        #     imgMN2O = drawFeaturesCorrespondance(imgl, kptsR, kptsROld, matchesN2O)


        oldCorrespondence = (matches, kptsL, descL, kptsR, descR)

        imgM = drawFeaturesCorrespondance(imgl, kptsL, kptsR, matches)
        imgl = cv2.drawKeypoints(imgl,kptsL, None)
        imgr = cv2.drawKeypoints(imgr,kptsR, None)
        cv2.imshow('left', imgl)
        cv2.imshow('right', imgr)
        cv2.imshow('matches-stereo', imgM)
        if imgMN2O != None:
            cv2.imshow('matches-nav', imgMN2O)
        key = cv2.waitKey(33) & 255
        if key == 27:
            break

    print(os.getcwd())
