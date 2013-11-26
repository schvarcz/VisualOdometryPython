from cv2 import *
import numpy as np
from dlt import *
import random
import csv


class VisualOdometry(object):
    def __init__(self,foco,distancia_cameras,confianca):

        self.foco = foco
        self.distancia_cameras = distancia_cameras
        self.confianca = confianca

        self.detector = FeatureDetector_create("SIFT")
        self.descriptor = DescriptorExtractor_create("SIFT")
        self.ste = StereoBM() #STEREO_BM_BASIC_PRESET,128,5)
        #self.ste = StereoSGBM(0,64,7,8*7*7,32*7*7,0,30,0)

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


        self.filecsv = csv.writer(file("posicao.csv",'w'),delimiter=';')

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
    
                    #print self.disps[self.nFrame-1].shape
                    if dist[i]<self.confianca and (0.5 < self.dists[self.nFrame-1][p[1],p[0]] < 0.6 ):
                        print self.dists[self.nFrame-1][p[1],p[0]]
                        line(self.frames[self.nFrame],oldP,p,(255,0,0))	
                        circle(self.frames[self.nFrame],p,3,(0,255,0),-1)

    def drawFeatures3(self):
        for pt in self.bestPoints:
            circle(self.frames[self.nFrame],(int(pt[0]),int(pt[1])),3,(255,0,255),-1)


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


    def computeProjectionError(self,coords,points,P):
        sumErr = 0.0
        for i in xrange(len(coords)):
            p=points[i]
            c=coords[i]
            c=np.matrix(c).T
            proj=P*c
            pix=[proj[0]/proj[2],proj[1]/proj[2]]
            sumErr += (pix[0]-p[0])*(pix[0]-p[0]) + (pix[1]-p[1])*(pix[1]-p[1])
        return sumErr/len(coords)


    def computeInliers(self,P,coords,points):
        inliers = []
        for i in xrange(len(coords)):
            p=points[i]
            c=coords[i]
            c=np.matrix(c).T
            proj=P*c
            pix=[proj[0]/proj[2],proj[1]/proj[2]]
            if (((pix[0]-p[0])*(pix[0]-p[0]) + (pix[1]-p[1])*(pix[1]-p[1])) < 2500):
                inliers.append(i)
        return inliers

    def computeRANSACDLT(self,coords,points):
        minError = 10000000
        best = None
        for i in range(1,50):
            # Computa o DLT com uma amostra de pontos
            indices = random.sample(xrange(len(coords)), 10)
            sampledCoords = [coords[j] for j in indices]
            sampledPoints = [points[j] for j in indices]
            k,r,t,P = DLT(sampledCoords,sampledPoints)

            # Computa o erro da projecao de todos os pontos considerando a matriz de projecao obtida
            err = self.computeProjectionError(coords,points,P)
            # print 'err({0}): {1}'.format(i,err)
            if err < minError:
                minError = err
                best = [k,r,t,P]
                self.bestPoints = sampledPoints

        print 'MeanError {0}!'.format(minError)

        inliers = self.computeInliers(best[3],coords,points)
        print 'total {0} inliers {1}'.format(len(coords),len(inliers))

        inliersCoords = [coords[j] for j in inliers]
        inliersPoints = [points[j] for j in inliers]

        for i in range(1,50):
            # Computa o DLT com uma amostra de pontos
            indices = random.sample(xrange(len(inliersCoords)), 10)
            sampledCoords = [inliersCoords[j] for j in indices]
            sampledPoints = [inliersPoints[j] for j in indices]
            k,r,t,P = DLT(sampledCoords,sampledPoints)

            # Computa o erro da projecao de todos os pontos considerando a matriz de projecao obtida
            err = self.computeProjectionError(inliersCoords,inliersPoints,P)
#            print 'err({0}): {1}'.format(i,err)
            if err < minError:
                minError = err
                best = [k,r,t,P]

        print 'MeanInliersError {0}!'.format(minError)

        return best


    def computeDistance(self,imgl,imgr):
        imglg, imgrg = cvtColor(imgl,cv.CV_RGBA2GRAY), cvtColor(imgr,cv.CV_RGBA2GRAY)
        disp = self.ste.compute(imglg, imgrg)/16.
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
                    x = (oldP[0]-6.601406e+02)*self.distancia_cameras/disp[oldP[1]][oldP[0]]
                    y = (oldP[1]-2.611004e+02)*self.distancia_cameras/disp[oldP[1]][oldP[0]]
                    z = dist[oldP[1]][oldP[0]]
                    coords.append([x,y,z,1.])
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
        #self.drawFeatures()

        self.matchingFeatures()

        #Computa a distancia em Z
        self.computeDistance(imgl,imgr)

        #Desenha correspondencia 2 a 2
        self.drawFeatures2()


        #X,Y,Z reais das features
        if self.nFrame != 0:
            coords, points = self.featuresCoordinates()
            
            k, r, t, P = DLT(coords,points)
            e = self.computeProjectionError(coords,points,P)

            print "k: ",k
            print "r: ",r
            print "t: ",t
            print "e: ",e

            k, r, t, P = self.computeRANSACDLT(coords,points)
            self.drawFeatures3()

            t = list(t.A.reshape(3))
            r = list(r.reshape(9))
            #err = list(err.A.reshape(1))
            self.filecsv.writerow(t+r)


