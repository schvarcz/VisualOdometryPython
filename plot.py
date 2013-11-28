# -*- coding:utf8 -*-
import csv
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def matrixRot(alpha,beta,gama):
    return asarray(
    [[cos(alpha)*cos(gama)-sin(alpha)*cos(beta)*sin(gama) ,  sin(alpha)*cos(gama)+cos(alpha)*cos(beta)*sin(gama) ,  sin(beta)*sin(gama)],
     [-cos(alpha)*sin(gama)-sin(alpha)*cos(beta)*cos(gama),  -sin(alpha)*sin(gama)+cos(alpha)*cos(beta)*cos(gama),  sin(beta)*cos(gama) ],
     [sin(beta)*sin(gama)                                 ,  -cos(alpha)*sin(beta)                               ,  cos(beta)]])

def matrixRot2(alpha,beta,gama):
    return matrix( 
    [ [+cos(beta)*cos(gama), -cos(beta)*sin(gama), +sin(beta)],
    [+sin(alpha)*sin(beta)*cos(gama)+cos(alpha)*sin(gama),-sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama),-sin(alpha)*cos(beta)],
    [-cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama), +cos(alpha)*sin(beta)*sin(gama)+sin(alpha)*cos(gama), +cos(alpha)*cos(beta)]])

f = plt.figure()
ax = f.add_subplot(111,projection='3d')
ax.set_xlim3d(-100,100)
ax.set_ylim3d(-100,100)

trans = [[],[],[]]
rots = [[],[],[]]
oldT = asarray([.6,.05,1.6])
oldR = asarray([0.,-.08,0.])

for line in csv.reader(file("posicao.csv"),delimiter=';'):
    line = [float(l) for l in line]

    t = asarray(line[0:3])
    r = asarray(line[3:])

    alpha, beta, gama = r

    alpha, beta, gama = oldR

    x, y, z = list(oldT)
    #t = asarray([y,z,x])

#    t = matrixRot(-alpha,-beta,-gama).dot(t)
    t = matrixRot2(alpha,beta,gama).I.dot(t).A1

    oldR += r
    oldT += t

    x, y, z = list(oldT)

    trans[0].append(-x)
    trans[1].append(-y)
    trans[2].append(-z)

    rots[0].append(alpha)
    rots[1].append(beta)
    rots[2].append(gama)

    print oldT


print oldT, oldR
#ax.plot(trans[1],trans[2],zeros(len(trans[0])))
ax.plot(trans[0],trans[2],trans[1])
#ax.plot(trans[0],trans[1],trans[2])
#ax.plot(trans[0],trans[2],zeros(len(trans[1])))




transReal = [[],[],[]]
tInicial = None
for line in csv.reader(file("dataset_libviso/insdata.txt"),delimiter=' '):
    t = asarray([float(l) for l in line[4:7]])
    t = matrixRot2(0,0,0.56).dot(t).A1
    if tInicial == None:
        tInicial = t
    t = t-tInicial
    for i in range(3):
        transReal[i].append(t[i])
    
ax.plot(transReal[0],transReal[1],zeros(len(transReal[2])))



transLibViso = [[],[],[]]
for line in csv.reader(file("saida.txt"),delimiter=';'):
    x,y,z = [float(l.split(' ')[-2]) for l in line][:3]
    
    transLibViso[0].append(x)
    transLibViso[1].append(y)
    transLibViso[2].append(z)


ax.plot(transLibViso[0],transLibViso[2],transLibViso[1])

ax.set_xlim3d(-60,100)
ax.set_ylim3d(-20,140)
ax.set_zlim3d(-60,100)


f1 = plt.figure()
plt.plot(trans[0],trans[2])
plt.plot(transReal[0],transReal[1])
plt.plot(transLibViso[0],transLibViso[2])
#plt.plot(rots[0])
#plt.plot(rots[1])
#plt.plot(rots[2])
plt.show()


