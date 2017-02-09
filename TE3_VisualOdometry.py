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

flann = cv2.FlannBasedMatcher(index_params,search_params)
featuredDetectorDescriptor = cv2.BRISK_create()


def cleanFeatures(matches, kptsL, descL, kptsR, descR):
    matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R = [], [], [], [], [], []
    idx = 0
    for m in matches:
        matches_r.append(cv2.DMatch(idx,idx,m.imgIdx,m.distance))
        kptsL_R.append(kptsL[m.queryIdx])
        kptsR_R.append(kptsR[m.trainIdx])
        descL_R.append(descL[m.queryIdx])
        descR_R.append(descR[m.trainIdx])
        dist_R.append(m.distance)
        idx += 1

    return matches_r, kptsL_R, descL_R, kptsR_R, descR_R, dist_R
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
                        if iMR2L.trainIdx == iML2R.queryIdx:
                            matches_r.append(iML2R)
                            matchFound = True
                            break
                else:
                    if matchesR2L[iML2R.trainIdx] == iML2R.queryIdx:
                        matches_r.append(iML2R)
                        matchFound = True
                if matchFound :
                    break
        elif mL2R != None:
            if type(matchesR2L[mL2R.trainIdx]) == list:
                for iMR2L in matchesR2L[mL2R.trainIdx]:
                    if iMR2L.trainIdx == mL2R.queryIdx:
                        matches_r.append(mL2R)
                        matchFound = True
                        break
            else:
                if matchesR2L[mL2R.trainIdx] == mL2R.queryIdx:
                    matches_r.append(mL2R)
                    matchFound = True
    return matches_r

def featuresCoordinates(kptsL, descL, kptsR, descR, dists):
    coords, kptsL_R, ktpsR_R = [], [], []
    for idx in xrange(len(kptsL)):

        dispX = kptsL[idx].pt[0] - kptsR[idx].pt[0]
        dispY = kptsL[idx].pt[1] - kptsR[idx].pt[1]

        if dits[idx] > confianca or dispX == 0 or abs(dispY) > 10:
            continue

        x = (kptsR[idx].pt[0]-centroProjecao[0])*distancia_cameras/dispX
        y = (kptsR[idx].pt[1]-centroProjecao[1])*distancia_cameras/dispX
        z = foco*distancia_cameras/dispX
        coords.append([x,y,z,1.])

    return coords, kptsL_R, kptsR_R

if __name__ == "__main__":
    for i in range(373):
        imgl, imgr = cv2.imread("{0}I1_{1:06d}.png".format(path,i)), cv2.imread("{0}I2_{1:06d}.png".format(path,i))

        kptsL = featuredDetectorDescriptor.detect(imgl)
        kptsL, descL = featuredDetectorDescriptor.compute(imgl,kptsL)
        kptsR = featuredDetectorDescriptor.detect(imgr)
        kptsR, descR = featuredDetectorDescriptor.compute(imgr,kptsR)

        matches = twoWayMatch(kptsL, descL, kptsR, descR)


        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           flags = 0)
        matches, kptsL, descL, kptsR, descR, dists = cleanFeatures(matches, kptsL, descL, kptsR, descR)

        print len(matches), matches[5].queryIdx
        print len(kptsL)
        # print len(kptsR[-1])

        img3 = cv2.drawMatchesKnn(imgl,kptsL,imgr,kptsR,matches, None)

        imgl = cv2.drawKeypoints(imgl,kptsL, None)
        imgr = cv2.drawKeypoints(imgr,kptsR, None)
        cv2.imshow('left', imgl)
        cv2.imshow('right', imgr)
        # cv2.imshow('matches', img3)
        key = cv2.waitKey(33) & 255
        if key == 27:
            break

    print(os.getcwd())
