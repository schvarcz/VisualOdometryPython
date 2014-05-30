# -*- coding: utf8 -*-
from cv2 import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm
from matplotlib.gridspec import GridSpec
import numpy as np

from visualodometry.VisualOdometry import VisualOdometry

#################
# Configurações #
#################
path = "2009_09_08_drive_0012/"
foco = 6.790081e+02
distancia_cameras = .57073824147 
confianca = 10000
pathSalvar = "salvar_1/"
centroProjecao = (6.639935e+02,2.452070e+02)
nframes = 2579

path = "dataset_libviso/"
foco = 6.452401e+02
distancia_cameras = .57073824147 
confianca = 10000
pathSalvar = "salvar_1/"
centroProjecao = (6.601406e+02,2.611004e+02)
nframes = 373


visual = VisualOdometry(foco,distancia_cameras,confianca,centroProjecao)

plt.ion()
f = plt.figure()
gs = GridSpec(2,2)
ax1 = plt.subplot(gs[0,1:])
ax2 = plt.subplot(gs[1,1:])
ax3 = plt.subplot(gs[:,0])
#ax4 = plt.subplot(gs[1,0])
#f, ((ax1,ax2),(ax3,ax4)) = subplots(2,2)

position = [[],[]]
angle = []
for i in range(nframes):
    imgl, imgr = imread("{0}I1_{1:06d}.png".format(path,i)), imread("{0}I2_{1:06d}.png".format(path,i))
    visual.compute(imgl,imgr)

    #Mostra resultados
    print "Features finais... ", i
#    imshow("Video",visual.currentFrame)

#    aux = visual.currentDist
#    aux[aux == float('inf')] = 0
#    aux[aux < 0.] = 0
#    aux= aux/(aux.max()*0.1)
#    imshow("Distancia",aux)
#    k = waitKey(33) & 255
#    if k == 27:
#        break;

    x,y,z = visual.position
    position[0].append(-x)
    position[1].append(-z)

    alpha, beta, gamma = visual.rotation
    angle.append(np.rad2deg(beta))

    ax1.imshow(visual.currentFrame)

    im1 = ax2.imshow(visual.currentDist, cmap = cm.gray)
    im1.set_clim(0.,100.)

    ax3.clear()
    ax3.plot(position[0],position[1],"b")
    ax3.set_xlabel("X Axis (m)")
    ax3.set_ylabel("Y Axis (m)")
    ax3.set_xlim(-10,80)
    ax3.set_ylim(-20,140)


#    ax4.clear()
#    ax4.plot(angle,"b")
#    ax4.set_xlabel("Frame")
#    ax4.set_ylabel("Angle")
#    ax4.set_xlim(0,nframes)
#    ax4.set_ylim(-180,180)

    f.canvas.draw()
    f.savefig("imgs/fig{0}".format(i))
    f.canvas.get_tk_widget().update() 

raw_input()
