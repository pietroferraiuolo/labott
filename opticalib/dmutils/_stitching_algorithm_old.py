import numpy as np
from opticalib.ground import zernike as zern
from opticalib import typings as t


def map_stitching(image_vector: t.CubeData, fullmask: t.ImageData, zern2fit: list[int]) -> t.ImageData:
    """
    Stitching algorithm. <br>
    Reference: `Otsubo, M., Okada, K., & Tsujiuchi, J. 1994, Optical Engineering, 33, 608`
    
    Parameters
    ----------
    image_vector : np.ndarray
        3D array of images to be stitched, shape (N, H, W).
    fullmask : np.ndarray
        2D array representing the full mask, shape (H, W).
    zern2fit : list
        List of Zernike indices to fit.

    Returns
    -------
    np.ndarray
        2D array of the stitched image after removing specified Zernike terms.
    """
    # Getting fundamental Dimensions
    N = image_vector.shape[0]
    MM = fullmask.copy()
    M = len(zern2fit)
    # Creating the base Zernike polynomial
    base = np.ma.masked_array(np.full(MM.shape, 2, dtype=float), MM)
    _, mat = zern.zernikeFit(base, zern2fit)
    temp = np.tri(M, dtype=float)
    p = np.array([zern.zernikeSurface(base, temp[i], mat) for i in range(M)])
    Qo = np.tile(p, (M, 1, 1))
    v_order = np.reshape(np.reshape(np.arange(M**2), (M, M)).T, (1, M**2))
    q = Qo * Qo[v_order[0]]
    # Computing P and Q Matrices
    # Main algorithm (and bottleneck)
    Q = []
    P = []
    for ii in range(N):
        for jj in range(N):
            print(f"{ii = :03d}/{N}, {jj = :03d}/{N}", end="\r", flush=True)
            mm = np.logical_or(image_vector[ii, :, :].mask, image_vector[jj, :, :].mask)
            if ii == jj:
                Q.append(np.zeros(M**2))
            else:
                Q.append(np.sum(q * (-1 * mm + 1), axis=(1, 2)))
            img = image_vector[ii, :, :] - image_vector[jj, :, :]
            P.append(np.array(np.nansum(p * (-1 * mm + 1) * img.data, axis=(1, 2))))
    # Reshaping P and Q Matrices
    P1 = np.reshape(np.array(P), (N, N, M))
    Pt = []
    for ii in range(N):
        temp = np.sum(P1[ii], axis=0)
        Pt.append(temp)
    PP = np.reshape(Pt, M * N)
    Q1 = np.reshape(np.array(Q), (N, N, M**2))
    QQ = np.reshape(Q1, (N, N, M, M))
    temp = np.vstack(
        [np.hstack([QQ[ii, jj, :, :] for ii in range(N)]) for jj in range(N)]
    )
    QQ = temp.copy()
    QD = np.zeros_like(QQ)
    for ii in range(N):
        temp = np.sum(Q1[ii], axis=0)
        temp = np.reshape(temp, (M, M))
        QD[M * ii : M * (ii + 1), M * ii : M * (ii + 1)] = -temp
    QF = QD + QQ
    X = np.linalg.lstsq(QF, PP, rcond=None)[0]  # Matrix inversion
    zzc = np.zeros_like(image_vector)
    c = np.reshape(X, (N, M))
    for ii in range(N):
        img = image_vector[ii, :, :].data
        mm = image_vector[ii, :, :].mask
        res = np.zeros_like(MM)
        for ki in range(M):
            res += p[ki] * c[ii, ki]
        zzc[ii, :, :] = np.ma.masked_array((img + res) * (-1 * mm + 1), mm)
    ZZ = np.ma.mean(zzc, axis=0)
    ZZ = zern.removeZernike(ZZ, zern2fit)
    return ZZ
