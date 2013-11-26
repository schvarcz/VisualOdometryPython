# -*- coding:utf8 -*-
import csv
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


f = plt.figure()
ax = f.add_subplot(111,projection='3d')
ax.set_xlim3d(-100,100)
ax.set_ylim3d(-100,100)
trans = [[],[],[]]
oldT = zeros(4)
oldT[3] = 1.
oldT = matrix(oldT).T

oldR = diag(ones(3))
for line in csv.reader(file("posicao.csv"),delimiter=';'):
    line = [float(l) for l in line]
    t = asarray(line[0:3])

    r = asarray(line[3:]).reshape(3,3)
    m = column_stack((r,t))
    m = row_stack((m,zeros(4)))
    m[3][3]=1.
    m = matrix(m)
    
    oldT= m.I*oldT
    
    #t = oldR.dot(t)
    #oldR = oldR.dot(r)
    #t += oldT
    #oldT = t


    x, y, z, h = list(oldT.T.A[0])

    trans[0].append(x)
    trans[1].append(y)
    trans[2].append(z)
#    for i in range(3):
#        trans[i].append(t[i])
    print oldT


print oldT
ax.plot(trans[0],trans[2],trans[1])





transReal = [[],[],[]]
tInicial = None
for line in csv.reader(file("/home/schvarcz/Dropbox/UFRGS - Mestrado/Robótica Avançada/Visual Odometry/disparity_stereo/dataset_libviso/insdata.txt"),delimiter=' '):
    t = asarray([float(l) for l in line[4:7]])
    print t
    if tInicial == None:
        tInicial = t
    t = t-tInicial
    for i in range(3):
        transReal[i].append(t[i])
    
#ax.plot(transReal[0],transReal[1],transReal[2])

ax.set_xlim3d(0,100)
ax.set_ylim3d(0,100)
ax.set_zlim3d(0,200)
f.show()
raw_input()
