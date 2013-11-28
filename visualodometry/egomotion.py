import numpy as np
import random

def computeRT(K,coords,points):
    tr=np.zeros(6)

    it = 0
    flag = True
    while(flag):
        newTr = IterationResidualJacobian(tr,K,coords,points)
        variacao = newTr - tr
        
        tr = newTr

        it += 1
        if (it == 20 or (variacao.max() < 1e-06)):
            flag = False

    return tr[:3],tr[3:]


def IterationResidualJacobian(tr,K,coords,points):
    focus = -K[0,0]
    cu = K[0,2]
    cv = K[1,2]

    # Initialization
    observation = points
    prediction  = [[1,1] for i in range(len(points))]
    residual    = [1 for i in range(len(points*2))]
    Jacobian    = [[1,1,1,1,1,1] for i in range(len(points)*2)]

    # Extract motion parameters
    rx = tr[0]; ry = tr[1]; rz = tr[2];
    tx = tr[3]; ty = tr[4]; tz = tr[5];

    # Precompute sine/cosine
    sx = np.sin(rx); cx = np.cos(rx); sy = np.sin(ry); cy = np.cos(ry); sz = np.sin(rz); cz = np.cos(rz);

    # Compute projection matrix (rot+trans)
    P = np.matrix([ [+cy*cz, -cy*sz, +sy, tx],
            [+sx*sy*cz+cx*sz,-sx*sy*sz+cx*cz,-sx*cy, ty],
            [-cx*sy*cz+sx*sz, +cx*sy*sz+sx*cz, +cx*cy, tz], [0., 0., 0., 1.]])

    # Derivatives of P
    drx = np.matrix([ [0., 0., 0., 0.],
            [+cx*sy*cz-sx*sz, -cx*sy*sz-sx*cz, -cx*cy, 0.],
            [+sx*sy*cz+cx*sz, -sx*sy*sz+cx*cz, -sx*cy, 0.], [0., 0., 0., 0.] ])
    dry = np.matrix([ [-sy*cz, +sy*sz, +cy, 0.],
            [+sx*cy*cz, -sx*cy*sz, +sx*sy, 0.],
            [-cx*cy*cz, +cx*cy*sz, -cx*sy, 0.], [0., 0., 0., 0.] ])
    drz = np.matrix([ [-cy*sz, -cy*cz, 0., 0.],
            [-sx*sy*sz+cx*cz, -sx*sy*cz-cx*sz, 0., 0.],
            [+cx*sy*sz+sx*cz, +cx*sy*cz-sx*sz, 0., 0.], [0., 0., 0., 0.] ])
    dtx = np.matrix([ [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    dty = np.matrix([ [0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.]])
    dtz = np.matrix([ [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])

    # For all points
    for i in range(len(coords)):
        pt = np.matrix(coords[i]).T

        # Project 3d point in current left coordinate system
        X = P*pt

        # Convert to pixel coordinates (project via K)
        prediction[i][0] = focus*X[0]/X[2]+cu # left u
        prediction[i][1] = focus*X[1]/X[2]+cv # left v
   
        # Compute residuals
        residual[2*i] = float(observation[i][0]-prediction[i][0])
        residual[2*i+1] = float(observation[i][1]-prediction[i][1])
        
        # Compute line of Jacobian
        q=drx*pt
        Jacobian[2*i][0] = float(focus*(q[0]*X[2]-X[0]*q[2] )/(X[2]**2))
        Jacobian[2*i+1][0] = float(focus*(q[1]*X[2]-X[1]*q[2] )/(X[2]**2))
        q=dry*pt
        Jacobian[2*i][1] = float(focus*(q[0]*X[2]-X[0]*q[2] )/(X[2]**2))
        Jacobian[2*i+1][1] = float(focus*(q[1]*X[2]-X[1]*q[2] )/(X[2]**2))
        q=drz*pt
        Jacobian[2*i][2] = float(focus*(q[0]*X[2]-X[0]*q[2] )/(X[2]**2))
        Jacobian[2*i+1][2] = float(focus*(q[1]*X[2]-X[1]*q[2] )/(X[2]**2))
        
        Jacobian[2*i][3] = float(focus/X[2])
        Jacobian[2*i+1][3] = 0

        Jacobian[2*i][4] = 0
        Jacobian[2*i+1][4] = float(focus/X[2])

        Jacobian[2*i][5] = float(-focus*X[0]/(X[2]**2))
        Jacobian[2*i+1][5] = float(-focus*X[1]/(X[2]**2))



    # Update parameters using residuals and Jacobian
    J = np.matrix(Jacobian)
    r = np.matrix(residual).T


    A = (J.T*J).I
    b = J.T*r

    newTr = tr + (A*b).A1

    return newTr

def computeError(K,r,t,coords,points):

    inliers = []

    # Extract motion parameters
    rx = r[0]; ry = r[1]; rz = r[2];
    tx = t[0]; ty = t[1]; tz = t[2];

    focus = -K[0,0]
    cu = K[0,2]
    cv = K[1,2]

    T=np.matrix(t).T

    # Precompute sine/cosine
    sx = np.sin(rx); cx = np.cos(rx); sy = np.sin(ry); cy = np.cos(ry); sz = np.sin(rz); cz = np.cos(rz);

    # Compute projection matrix (rot+trans)
    R = np.matrix([ [+cy*cz, -cy*sz, +sy],
            [+sx*sy*cz+cx*sz,-sx*sy*sz+cx*cz,-sx*cy],
            [-cx*sy*cz+sx*sz, +cx*sy*sz+sx*cz, +cx*cy]])

    sumErr = 0.0
    for i in xrange(len(coords)):
        p=points[i]
        c=coords[i][:3]
        c=np.matrix(c).T

        proj=(R*c+T)
#        print '##c:{0} p:{1}##'.format(c,proj)
        pix=[focus*proj[0]/proj[2]+cu,focus*proj[1]/proj[2]+cv]
        err = (pix[0]-p[0])*(pix[0]-p[0]) + (pix[1]-p[1])*(pix[1]-p[1])
        if (err < 50):
            inliers.append(i)

#        print '##err:{0} {1}-{2}##'.format(err,pix,p)
        sumErr += err
    return sumErr/len(coords), inliers

def estimateMotion(K,coords,points):

    minError = 10000000
    best = None

#    computeObservations(p_matched,active);
#    computeResidualsAndJacobian(tr,active);

    # Executa o RANSAC com 50 iteracoes
    for i in range(1,25):
        # Seleciona amostra de pontos
        indices = random.sample(xrange(len(coords)), 6)
        sampledCoords = [coords[j] for j in indices]
        sampledPoints = [points[j] for j in indices]
        r,t = computeRT(K,sampledCoords,sampledPoints)

        # Computa o erro da projecao de todos os pontos considerando a matriz de projecao obtida
        err,inliers = computeError(K,r,t,coords,points)
#        print 'err({0}): {1}'.format(i,err)
        if err < minError:
            minError = err
            best = [r,t,inliers]
#            self.bestPoints = sampledPoints

    print 'MeanError {0}!'.format(minError)

    inliers = best[2]
#    inliers = self.computeInliers(best[3],coords,points)
    print 'total {0} inliers {1}'.format(len(coords),len(inliers))

    minError = 10000000

    inliersCoords = [coords[j] for j in inliers]
    inliersPoints = [points[j] for j in inliers]

    # Repete RANSAC usando somente inliers
    for i in range(1,25):
        # Seleciona amostra de pontos
        indices = random.sample(xrange(len(inliersCoords)), 6)
        sampledCoords = [inliersCoords[j] for j in indices]
        sampledPoints = [inliersPoints[j] for j in indices]
        r,t = computeRT(K,sampledCoords,sampledPoints)

        # Computa o erro da projecao de todos os pontos considerando a matriz de projecao obtida
        err,inliers = computeError(K,r,t,inliersCoords,inliersPoints)
#        print 'err({0}): {1}'.format(i,err)
        if err < minError:
            minError = err
            best = [r,t,inliers]
#            self.bestPoints = sampledPoints
    print 'MeanInliersError {0}!'.format(minError)

    return [best[0],best[1]]

