from cv2 import *
import numpy as np
from dlt import *


class VisualOdometry(object):
    def __init__(self,foco,distancia_cameras,confianca):

        self.foco = foco
        self.distancia_cameras = distancia_cameras
        self.confianca = confianca

        self.detector = FeatureDetector_create("SIFT")
        self.descriptor = DescriptorExtractor_create("SIFT")
        self.ste = StereoBM()

        range_frames = 2

        self.sds = [None for i in range(range_frames)]
        self.skps = [None for i in range(range_frames)]
        self.flanns = [None for i in range(range_frames)]
        self.frames = [None for i in range(range_frames)]
        self.disps = [None for i in range(range_frames)]
        self.dists = [None for i in range(range_frames)]

        self.idxs = [None for i in range(range_frames)]
        self.featuresDists = [None for i in range(range_frames)]
        self.idxClean = None
        self.distClean = None

        self.nFrame = -1
        self.flann_params = dict(algorithm=1, trees=1)
        self.centro_imagem = None

    @property
    def currentDist(self):
        return self.dists[self.nFrame]

    @property
    def currentDisp(self):
        return self.disps[self.nFrame]

    @property
    def currentFrame(self):
        return self.frames[self.nFrame]

    def findCurrentFeatures(self):
        index = self.nFrame
        img = self.frames[index]

        skp = self.detector.detect(img)
        skp, sd = self.descriptor.compute(img,skp)
        flann = flann_Index(sd, self.flann_params)

        self.skps[index], self.sds[index], self.flanns[index] = skp,sd,flann


    def drawFeatures(self):
        skp = self.skps[self.nFrame]
        for i in xrange(len(skp)):            
            circle(self.frames[self.nFrame],(int(skp[i].pt[0]),int(skp[i].pt[1])),3,(0,0,255),-1)


    def drawFeatures2(self):
        if self.nFrame != 0:
            skpOld = self.skps[self.nFrame-1]
            skp = self.skps[self.nFrame]
            idx = self.idxs[self.nFrame-1]
            dist = self.featuresDists[self.nFrame-1]

            for i in xrange(len(idx)):

                if idx[i] != None:
                    oldP = (int(skpOld[i].pt[0]),int(skpOld[i].pt[1]))
                    p = (int(skp[idx[i]].pt[0]),int(skp[idx[i]].pt[1]))

                    if dist[i]<self.confianca:
                        line(self.frames[self.nFrame],oldP,p,(255,0,0))	
                        circle(self.frames[self.nFrame],p,3,(0,255,0),-1)


    def matchingFeatures(self):
        #Faz a correspondencia 2 a 2
        index = self.nFrame

        if index != 0:
            sd, sdOld = self.sds[index], self.sds[index-1]
            flann, flannOld = self.flanns[index], self.flanns[index-1]
            self.idxs[index-1], self.featuresDists[index-1] = self.twoWayMatch(sdOld, flannOld, sd, flann)

        #Faz a correspondencia 2 a 2 do ultimo com o primeiro
        if index == len(self.frames)-1:
            sd, sdOld = self.sds[index], self.sds[0]
            flann, flannOld = self.flanns[index], self.flanns[0]
            self.idxs[index], self.featuresDists[index] = self.twoWayMatch(sdOld, flannOld, sd, flann)


    def twoWayMatch(self, sdOld, flannOld, sd, flann):
        idxOld, distOld = flann.knnSearch(sdOld, 1, params={})
        idx, dist = flannOld.knnSearch(sd, 1, params={})
        idxOld, distOld = list(idxOld), list(distOld)

        for i in xrange(len(idxOld)):
            if type(idxOld[i]) == list or type(idxOld[i]) == tuple:
                for j in idxOld[i]:
                    if i in idx[j]:
                        distOld[i] = distOld[i][idxOld[i].index(j)]
                        idxOld[i] = j
                if type(idxOld[i]) != int:
                    idxOld[i] = None
                    distOld[i] = None
            else:
                if i != idx[idxOld[i]]:
                    idxOld[i] = None
                    distOld[i] = None

        return idxOld, distOld


    def computeDistance(self,imgl,imgr):
        imglg, imgrg = cvtColor(imgl,cv.CV_RGBA2GRAY), cvtColor(imgr,cv.CV_RGBA2GRAY)
        disp = self.ste.compute(imglg, imgrg)
        
        dist = self.foco*self.distancia_cameras/disp
        self.disps[self.nFrame], self.dists[self.nFrame] = disp, dist

    def featuresCoordinates(self):
        coords = []
        points = []
        
        index = self.nFrame
        idx = self.idxs[index-1]
        skpOld, skp = self.skps[index-1], self.skps[index]
        featureDist = self.featuresDists[index-1]
        disp, dist = self.currentDisp, self.currentDist

        for i in xrange(len(idx)):
            if idx[i] != None:
                oldP = (int(skpOld[i].pt[0]),int(skpOld[i].pt[1]))
                p = (int(skp[idx[i]].pt[0]),int(skp[idx[i]].pt[1]))
                if featureDist[i]<self.confianca and disp[oldP[1]][oldP[0]] > 0 :
                    x = (oldP[0]-self.centro_imagem[0])*self.distancia_cameras/disp[oldP[1]][oldP[0]]
                    y = (oldP[1]-self.centro_imagem[1])*self.distancia_cameras/disp[oldP[1]][oldP[0]]
                    z = dist[oldP[1]][oldP[0]]
                    coords.append([x,y,z])
                    points.append(p)

        return coords, points


    def compute(self,imgl,imgr):
        if self.centro_imagem == None:
            self.centro_imagem = imgl.shape[0]/2.,imgl.shape[1]/2.


        #Move a janela das features
        if (self.nFrame < len(self.frames)-1):
            self.nFrame += 1
        else:
            for i in range(1,len(self.frames)):
                self.frames[i-1] = self.frames[i]
                self.skps[i-1] = self.skps[i]
                self.sds[i-1] = self.sds[i]
                self.flanns[i-1] = self.flanns[i]
                self.idxs[i-1] = self.idxs[i]
                self.featuresDists[i-1] = self.featuresDists[i]

        #acha as features para a imagem atual
        self.frames[self.nFrame] = imgl
        self.findCurrentFeatures()

        #Desenha os features encontrados
        self.drawFeatures()

        self.matchingFeatures()


        #Desenha correspondencia 2 a 2
        self.drawFeatures2()

        #Computa a distancia em Z
        self.computeDistance(imgl,imgr)

        #X,Y,Z reais das features
        if self.nFrame != 0:
            coords, points = self.featuresCoordinates()
            
            k,r,t = DLT(coords,points)

            print "k: ",k
            print "r: ",r
            print "t: ",t


