# -*- coding: utf8 -*-
from cv2 import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm
import numpy as np
from scipy.linalg import qr

def rq(A): 
    Q,R = qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T 
    return R[:,::-1],Q[::-1,:]

#################
# Configurações #
#################
path = "dataset_libviso/"
foco = 8.941981e+02
distancia_cameras =.57073824147 
confianca = 5000
pathSalvar = "salvar_1/"


detector = FeatureDetector_create("SIFT")
descriptor = DescriptorExtractor_create("SIFT")
ste = StereoBM()


range_frames = 2
sds = [None for i in range(range_frames)]
skps = [None for i in range(range_frames)]
flanns = [None for i in range(range_frames)]
frames = [None for i in range(range_frames)]

idxs = [None for i in range(range_frames)]
dists = [None for i in range(range_frames)]
idxClean = None
distClean = None

tempo = 0 
nFrame = 0
flann_params = dict(algorithm=1, trees=1)

disp = None
dist = None
ttotal = None
for i in range(373):
    imgl, imgr = imread("{0}I1_{1:06d}.png".format(path,i)), imread("{0}I2_{1:06d}.png".format(path,i))
    imglg, imgrg = cvtColor(imgl,cv.CV_RGBA2GRAY), cvtColor(imgr,cv.CV_RGBA2GRAY)


    #acha as features para a imagem atual
    frames[nFrame] = imgl
    skps[nFrame] = detector.detect(imgl)
    skps[nFrame], sds[nFrame] = descriptor.compute(imgl,skps[nFrame])
    flanns[nFrame] = flann_Index(sds[nFrame], flann_params)

    #Desenha os features encontrados
    for i in xrange(len(skps[nFrame])):            
        circle(imgl,(int(skps[nFrame][i].pt[0]),int(skps[nFrame][i].pt[1])),3,(0,0,255),-1)

    #Faz a correspondencia 2 a 2
    if nFrame != 0:
        idx1, dist1= flanns[nFrame].knnSearch(sds[nFrame-1], 1, params={})
        idx2, dist2 = flanns[nFrame-1].knnSearch(sds[nFrame], 1, params={})
        idxs[nFrame-1], dists[nFrame-1] = list(idx1),list(dist1)
        for i in xrange(len(idx1)):
            if type(idx1[i]) == list or type(idx1[i]) == tuple:
                for j in idx1[i]:
                    if i in idx2[j]:
                        dists[nFrame-1][i] = dists[nFrame-1][i][idxs[nFrame-1][i].index(j)]
                        idxs[nFrame-1][i] = j
                if type(idx1[i]) != int:
                    idxs[nFrame-1][i] = None
                    dists[nFrame-1][i] = None
            else:
                if i != idx2[idx1[i]]:
                    idxs[nFrame-1][i] = None
                    dists[nFrame-1][i] = None

    #Faz a correspondencia 2 a 2 do ultimo com o primeiro
    if nFrame == len(frames)-1:
        idx1, dist1= flanns[nFrame].knnSearch(sds[0], 1, params={})
        idx2, dist2 = flanns[0].knnSearch(sds[nFrame], 1, params={})
        idxs[nFrame], dists[nFrame] = list(idx1),list(dist1)
        for i in xrange(len(idx1)):
            if type(idx1[i]) == list or type(idx1[i]) == tuple:
                for j in idx1[i]:
                    if i in idx2[j]:
                        dists[nFrame][i] = dists[nFrame][i][idxs[nFrame][i].index(j)]
                        idxs[nFrame][i] = j
                if type(idx1[i]) != int:
                    idxs[nFrame][i] = None
                    dists[nFrame][i] = None
            else:
                if i != idx2[idx1[i]]:
                    idxs[nFrame][i] = None
                    dists[nFrame][i] = None



    #Desenha correspondencia 2 a 2
    if nFrame != 0:
        for i in xrange(len(idxs[nFrame-1])):
            if idxs[nFrame-1][i] != None:
                #print "{0}: {1}".format(i,idxs[nFrame][i])
                oldP = (int(skps[nFrame-1][i].pt[0]),int(skps[nFrame-1][i].pt[1]))
                p = (int(skps[nFrame][idxs[nFrame-1][i]].pt[0]),int(skps[nFrame][idxs[nFrame-1][i]].pt[1]))
                if dists[nFrame-1][i]<confianca:
                    line(imgl,oldP,p,(255,0,0))	
                    #circle(imgl,oldP,3,(255,0,0),-1)	
                    circle(imgl,p,3,(0,255,0),-1)

    #X,Y,Z reais das features
    coords = []
    points = []
    centro_imagem = imgl.shape[0]/2.,imgl.shape[1]/2.
    print centro_imagem
    if nFrame != 0:
        for i in xrange(len(idxs[nFrame-1])):
            if idxs[nFrame-1][i] != None:
                oldP = (int(skps[nFrame-1][i].pt[0]),int(skps[nFrame-1][i].pt[1]))
                p = (int(skps[nFrame][idxs[nFrame-1][i]].pt[0]),int(skps[nFrame][idxs[nFrame-1][i]].pt[1]))
                if dists[nFrame-1][i]<confianca and disp[oldP[1]][oldP[0]] > 0 :
                    x = (oldP[0]-centro_imagem[0])*.57073824147/disp[oldP[1]][oldP[0]]
                    y = (oldP[1]-centro_imagem[1])*.57073824147/disp[oldP[1]][oldP[0]]
                    z = dist[oldP[1]][oldP[0]]
                    coords.append([x,y,z])
                    points.append(p)
                    print "x: ",x,"y: ",y,"z: ",z
        
        
        A = []
        for i in range(len(coords)):
            A.append([coords[i][0],coords[i][1],coords[i][2], 1, 0,0,0,0, -points[i][0]*coords[i][0], -points[i][0]*coords[i][1], -points[i][0]*coords[i][2]])
            A.append([0,0,0,0, coords[i][0],coords[i][1],coords[i][2], 1 , -points[i][1]*coords[i][0], -points[i][1]*coords[i][1], -points[i][1]*coords[i][2]])

        A = np.matrix(A)

        b = np.matrix(np.asarray(points).reshape(len(points)*2)).T

        print "\n\n\n\nDLT: "
        P = np.row_stack((np.linalg.solve(A.T*A,A.T*b),[1.])).reshape(3,4   )
        print "P: ",P
        r,q = rq(P[:,:3])
        s = np.diag([np.sign(-r[0][0]),np.sign(-r[1][1]),np.sign(r[2][2])])
        k = np.dot(r,s)
        r = np.dot(s,q)
        t = np.matrix(k).I*P[:,3]
        if ttotal == None:
            ttotal = t
        else:
            ttotal += t
        print "k: ",k
        print "r: ",r
        print "t: ",t
        print "ttotal: ",ttotal

    #Computa a distancia em Z
    disp = ste.compute(imglg, imgrg)

    
    dist = foco*distancia_cameras/disp
    distCor = merge((dist,dist,dist))

    #Mostra resultados
    imshow("Distancia",distCor)
    print "Features finais... "
    imshow("Video",imgl)
    k = waitKey(33) & 255
    if k == 27:
        break;


    #Move a janela das features
    if (nFrame < len(frames)-1):
        nFrame += 1
    else:
        for i in range(1,len(frames)):
            frames[i-1] = frames[i]
            skps[i-1] = skps[i]
            sds[i-1] = sds[i]
            flanns[i-1] = flanns[i]
            idxs[i-1] = idxs[i]
            dists[i-1] = dists[i]



