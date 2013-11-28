# -*- coding:utf8 -*-
import csv
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def matrixRot(alpha,beta,gama):
    return matrix( 
    [ [+cos(beta)*cos(gama), -cos(beta)*sin(gama), +sin(beta)],
    [+sin(alpha)*sin(beta)*cos(gama)+cos(alpha)*sin(gama),-sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama),-sin(alpha)*cos(beta)],
    [-cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama), +cos(alpha)*sin(beta)*sin(gama)+sin(alpha)*cos(gama), +cos(alpha)*cos(beta)]])


trans = [[],[],[]]
rots = [[],[],[]]
oldT = asarray([.6,.05,1.6]) #posição da camera em relação ao gps
oldR = asarray([0.,-.08,0.]) #rotação da camera em relação ao gps

for line in csv.reader(file("posicao.csv"),delimiter=';'):
    line = [float(l) for l in line]

    t = asarray(line[0:3])
    r = asarray(line[3:])

    alpha, beta, gama = oldR

    t = matrixRot(alpha,beta,gama).I.dot(t).A1

    oldR += r
    oldT += t

    x, y, z = list(oldT)

    trans[0].append(-x)
    trans[1].append(-y)
    trans[2].append(-z)

    rots[0].append(alpha)
    rots[1].append(beta)
    rots[2].append(gama)



transGPS = [[],[],[]]
tInicial = None
for line in csv.reader(file("dataset_libviso/insdata.txt"),delimiter=' '):
    t = asarray([float(l) for l in line[4:7]])
    t = matrixRot(0,0,0.56).dot(t).A1 #Alguma rotação do GPS, por algum motivo.
    if tInicial == None:
        tInicial = t
    t = t-tInicial
    for i in range(3):
        transGPS[i].append(t[i])
    


transLibViso = [[],[],[]]
for line in csv.reader(file("saida.txt"),delimiter=';'):
    x,y,z = [float(l.split(' ')[-2]) for l in line][:3]
    
    transLibViso[0].append(x)
    transLibViso[1].append(y)
    transLibViso[2].append(z)



#Inicia as plotagens paths

#Path 3D
f = plt.figure()
ax = f.add_subplot(111,projection='3d')

ax.plot(trans[0],trans[2],trans[1],label="Nosso")
ax.plot(transGPS[0],transGPS[1],zeros(len(transGPS[2])),label="GPS")
ax.plot(transLibViso[0],transLibViso[2],transLibViso[1],label="LibViso")

ax.set_xlim3d(-60,100)
ax.set_ylim3d(-20,140)
ax.set_zlim3d(-60,100)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.legend()


#Path 2D
f1 = plt.figure()
plt.plot(trans[0],trans[2],label="Nosso")
plt.plot(transGPS[0],transGPS[1],label="GPS")
plt.plot(transLibViso[0],transLibViso[2],label="LibViso")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend()





#Erro dos path's

trans = zip(*trans)
transGPS = zip(*transGPS)
transLibViso = zip(*transLibViso)
disLibGPS = []
disLibNosso = []
disGPSNosso = []

def distance(pt1,pt2):
    return sqrt(((pt1-pt2)**2).sum())

for i in xrange(len(trans)):
    pt = asarray(trans[i])
    ptl = asarray(transLibViso[i])
    ptr = asarray(transGPS[i])
    d1 = distance(pt,ptl)
    d2 = distance(pt,ptr)
    d3 = distance(ptl,ptr)
    disLibNosso.append(d1)
    disGPSNosso.append(d2)
    disLibGPS.append(d3)

media = zeros(3)
for i in xrange(len(disLibNosso)):
    media[0] += disLibNosso[i]
    media[1] += disGPSNosso[i]
    media[2] += disLibGPS[i]

media[0] /= len(disLibNosso)
media[1] /= len(disGPSNosso)
media[2] /= len(disLibGPS)


variancia = zeros(3)
for i in xrange(len(disGPSNosso)):

    variancia[0] += (disLibNosso[i]-media[0])**2
    variancia[1] += (disGPSNosso[i]-media[1])**2
    variancia[2] += (disLibGPS[i]-media[2])**2

variancia[0] /= len(disLibNosso)
variancia[1] /= len(disGPSNosso)
variancia[2] /= len(disLibGPS)

#plotagem
f2 = plt.figure()
plt.plot(disLibNosso,label='Erro Nosso-LibViso')
plt.plot(disGPSNosso,label='Erro Nosso-GPS')
plt.plot(disLibGPS,label='Erro GPS-LibViso')
plt.xlabel("Frame")
plt.ylabel("Erro (m)")
plt.legend()

#Media e variancia
print "Média: \n",media
print "Variancia: \n",variancia

plt.show()


