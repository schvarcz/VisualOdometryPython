
import numpy as np
from scipy.linalg import rq

def DLT(coords,points):
    A = []
    for i in range(len(coords)):
        A.append([coords[i][0],coords[i][1],coords[i][2], 1, 0,0,0,0, -points[i][0]*coords[i][0], -points[i][0]*coords[i][1], -points[i][0]*coords[i][2]])
        A.append([0,0,0,0, coords[i][0],coords[i][1],coords[i][2], 1 , -points[i][1]*coords[i][0], -points[i][1]*coords[i][1], -points[i][1]*coords[i][2]])

    A = np.matrix(A)    
    b = np.matrix(np.asarray(points).reshape(len(points)*2)).T

    P = np.row_stack((np.linalg.solve(A.T*A,A.T*b),[1.])).reshape(3,4)
    return decompCameraMatrix(P)

def decompCameraMatrix(P):    
    r,q = rq(P[:,:3])
    s = np.diag([np.sign(-r[0][0]),np.sign(-r[1][1]),np.sign(r[2][2])])
    k = np.dot(r,s)
    r = np.dot(s,q)
    t = np.matrix(k).I*P[:,3]
    return k,r,t,P
