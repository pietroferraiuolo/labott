## zz è un cubo con tuttle le immagini da stitchare, già nelle loro posizioni giuste
## xx e yy sono le matrici delle coordinate
## idr zernike da usare per correggere
## idx zernike da rimuovere nell'immagine finale

import numpy as np
from opticalib.ground import zernike as zern
from typing import Callable, Any

import time
import functools
import gc


def timer(func: Callable) -> Callable:
    """
    Simple timing decorator for functions.
    
    Parameters
    ----------
    func : Callable
        Function to be timed.
        
    Returns
    -------
    Callable
        Wrapped function that prints execution time.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@timer
def map_stitching(zz, fullmask, idr):
    """
    
    Parameters
    ----------
    zz : np.ndarray
        3D array of images to be stitched, shape (N, H, W).
    fullmask : np.ndarray
        2D array representing the full mask, shape (H, W).
    idr : list
        List of Zernike indices to fit.
    idx : list
        List of Zernike indices to remove from the final image.
    """
# zz = imgvecout
# idr = zernike to fit
# idx = zernike to remove

    N = zz.shape[0]  # Number of images
    MM = fullmask.copy()
    M = len(idr)

#   p = [ZernikeCalc(i, 1, MM, 'noll') for i in idr]  # Create base matrices

    base = np.ma.masked_array(np.full(MM.shape, 2, dtype=float), MM)
    _, mat = zern.zernikeFit(base, idr)
    
    temp = np.tri(M, dtype=float)
    p = np.array([zern.zernikeSurface(base, temp[i], mat) for i in range(M)])

    # p = []
    # temp = np.zeros(M)
    # for ii in range(M):
        # temp[ii]=1
        # surf = zern.zernikeSurface(base, temp, mat)
        # p.append(surf)

    Qo = np.tile(p, (M, 1,1))
#    Qv = np.tile(np.array(p).T, (1, M)) # matrici trasposte che non ci piacciono
# matrice diagonale a blocchi

    v_order = np.reshape(
        np.reshape(
            np.arange(M**2),(M,M)
            ).T,
        (1,M**2)
    )

    # q = []
    # for ii in range(M**2): # Compute q matrices
    #     qo = Qo[ii]
    #     qv = Qo[v_order[0,ii]]
    #     q.append(qo*qv)
    q = Qo * Qo[v_order[0]]

    Q = []
    P = []
    for ii in range(N):
        for jj in range(N):
            print(f"{ii = :03d}/{N}, {jj = :03d}/{N}", end='\r', flush=True)
            mm = np.logical_or(zz[ii, :, :].mask, zz[jj, :, :].mask)
#            idx = np.argwhere(mm==True)
            if ii == jj:
               # Q[(ii, jj)] = np.zeros((M, M))
                Q.append(np.zeros(M**2))
            else:
#                 Qn = []
#                 for ki in range(M**2):
# #                    temp = np.ma.masked_array(q[ki], mm)
#                     temp = q[ki]*(-1*mm+1)
#                     #temp[idx] = 0.0
#                     Qn.append(np.sum(temp[:]))

               # for ki in range(M):
               #     for kj in range(M):
               #         Qn[ki, kj] = np.sum(q[ki][kj][mm])
                Q.append(np.sum(q * (-1*mm+1), axis=(1,2)))

            img = zz[ii, :, :] - zz[jj, :, :]
#             Pn = []
#             for ki in range(M):
#                 temp2 = p[ki]*(-1*mm+1)
# #                temp2[idx] = 0.0
# #                np.ma.masked_array(p[ki],mm).data
#                 Pn.append(np.array(np.nansum(temp2*img.data)))
            P.append(np.array(np.nansum(p*(-1*mm+1)*img.data, axis=(1,2))))

    #P = np.array(P)
    P1 = np.reshape(np.array(P),(N,N,M))

    Pt = []
    for ii in range(N):
#        temp = np.sum(P1[:,ii],axis=0)
        temp = np.sum(P1[ii],axis=0)
        Pt.append(temp)

    PP = np.reshape(Pt,M*N)

    #Q = np.array(Q)
    Q1 = np.reshape(np.array(Q),(N,N,M**2))
    QQ = np.reshape(Q1,(N,N,M,M))

    temp = np.vstack([np.hstack([QQ[ii, jj,:,:] for ii in range(N)]) for jj in range(N)]) #N-1

    QQ = temp.copy()
#    QQ = np.vstack([np.hstack([Q[(ii, jj)] for jj in range(N-1)]) for ii in range(N-1)])

    QD = np.zeros_like(QQ)
    for ii in range(N):
        temp = np.sum(Q1[ii],axis=0)
        temp = np.reshape(temp,(M,M))
        QD[M*ii:M*(ii+1), M*ii:M*(ii+1)] = -temp


#    for ii in range(N-1): #N-1
#        tmp = np.zeros((M, M))
#        for k in range(N):
#            tmp += Q[(ii, k)]
#        QD[M*ii:M*(ii+1), M*ii:M*(ii+1)] = -tmp

    QF = QD + QQ

    X = np.linalg.lstsq(QF, PP, rcond=None)[0]  # Matrix inversion

    zzc = np.zeros_like(zz)
    c = np.reshape(X,(N, M)) #N-1!!!!

    for ii in range(N):#N-1!!!!
        img = zz[ii, :, :].data
        mm = zz[ii, :, :].mask
        res = np.zeros_like(MM)
        for ki in range(M):
            res += p[ki] * c[ii, ki]
        # tmp = img + res
        # tmp = tmp*(-1*mm+1)
        # tmp = np.ma.masked_array((img+res)*(-1*mm+1),mm)
#        tmp[~mm] = 0
        zzc[ii, :, :] = np.ma.masked_array((img+res)*(-1*mm+1),mm)

#    zzc[:, :, N-1] = zz[:, :, N-1]

    ZZ = np.ma.mean(zzc,axis=0)
#    ZZ = np.sum(zzc, axis=0) / np.sum(np.isfinite(zz), axis=0)
    ZZ = zern.removeZernike(ZZ,idr)

    return ZZ
