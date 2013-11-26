# -*- coding: utf8 -*-
from cv2 import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm
import numpy as np

from visualodometry.VisualOdometry import VisualOdometry

#################
# Configurações #
#################
path = "dataset_libviso/"
foco = 6.452401e+02
distancia_cameras = .57073824147 
confianca = 10000
pathSalvar = "salvar_1/"


visual = VisualOdometry(foco,distancia_cameras,confianca)

#plt.ion()
#f, (ax1,ax2) = subplots(2,1)

for i in range(373):
    imgl, imgr = imread("{0}I1_{1:06d}.png".format(path,i)), imread("{0}I2_{1:06d}.png".format(path,i))
    visual.compute(imgl,imgr)

    #Mostra resultados
    print "Features finais... ", i
    imshow("Video",visual.currentFrame)

    aux = visual.currentDist
    aux[aux == float('inf')] = 0
    aux[aux < 0.] = 0
    #print aux.min(),aux.max()
    aux= aux/(aux.max()*0.10)
    #print aux.min(),aux.max()
    imshow("Distancia",aux)
    k = waitKey(33) & 255
    if k == 27:
        break;

#    ax1.imshow(visual.currentFrame)
#    ax2.imshow(visual.currentDist)
#    f.canvas.draw()
#    f.canvas.get_tk_widget().update() 

