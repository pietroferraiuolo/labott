import numpy as np
from typing import List, Tuple
from opticalib.ground import zernike as zern


def map_stitching(
    zz: np.ndarray, 
    fullmask: np.ndarray, 
    idr: List[int]
) -> np.ndarray:
    """
    Stitch multiple images using Zernike polynomial fitting.
    
    Parameters
    ----------
    zz : np.ndarray
        3D array of images to be stitched, shape (N, H, W).
    fullmask : np.ndarray
        2D array representing the full mask, shape (H, W).
    idr : list
        List of Zernike indices to fit.
        
    Returns
    -------
    np.ndarray
        Stitched and corrected image.
    """
    N = zz.shape[0]  # Number of images
    M = len(idr)
    MM = fullmask.copy()
    
    # Precompute Zernike basis functions
    pippo = np.ones(np.shape(MM)) * 2
    base = np.ma.masked_array(pippo, MM)
    _, mat = zern.zernikeFit(base, idr)
    
    # Vectorized computation of Zernike surfaces
    p = np.zeros((M,) + MM.shape)
    for ii in range(M):
        temp = np.zeros(M)
        temp[ii] = 1
        p[ii] = zern.zernikeSurface(base, temp, mat)
    
    # Precompute q matrices more efficiently
    Qo = np.tile(p, (M, 1, 1))
    v_temp = np.arange(M**2)
    m_temp = np.reshape(v_temp, (M, M))
    v_order = np.reshape(m_temp.T, M**2)
    
    # Vectorized q computation
    q = Qo[np.arange(M**2)] * Qo[v_order]
    
    # Precompute all masks
    masks = np.zeros((N, N) + zz.shape[1:], dtype=bool)
    for ii in range(N):
        masks[ii] = np.logical_or(zz[ii].mask[None, :, :], zz.mask)
    
    # Vectorized Q and P computation
    Q = np.zeros((N, N, M**2))
    P = np.zeros((N, N, M))
    
    # Create mask for valid (non-masked) pixels
    mask_weights = (~masks).astype(np.float32)
    
    # Vectorized Q computation
    for ii in range(N):
        for jj in range(N):
            if ii == jj:
                Q[ii, jj] = 0
            else:
                # Broadcast multiplication and sum
                Q[ii, jj] = np.sum(
                    q * mask_weights[ii, jj][None, :, :], 
                    axis=(1, 2)
                )
    
    # Vectorized P computation using broadcasting
    img_diffs = zz[:, None] - zz[None, :]  # Shape: (N, N, H, W)
    
    for ii in range(N):
        for jj in range(N):
            # Vectorized computation across all M elements
            P[ii, jj] = np.sum(
                p * mask_weights[ii, jj][None, :, :] * 
                img_diffs[ii, jj].data[None, :, :], 
                axis=(1, 2)
            )
    
    # Reshape and compute system matrices more efficiently
    P_reshaped = P.reshape(N, N, M)
    PP = np.sum(P_reshaped, axis=1).reshape(M * N)
    
    Q_reshaped = Q.reshape(N, N, M, M)
    
    # Create block matrix more efficiently using advanced indexing
    block_indices = np.arange(N)[:, None] * M + np.arange(M)[None, :]
    QQ = np.zeros((N * M, N * M))
    
    for ii in range(N):
        for jj in range(N):
            QQ[block_indices[ii][:, None], block_indices[jj][None, :]] = Q_reshaped[ii, jj]
    
    # Diagonal blocks
    QD = np.zeros_like(QQ)
    Q_diagonal_sums = -np.sum(Q_reshaped, axis=1)
    
    for ii in range(N):
        start_idx = ii * M
        end_idx = (ii + 1) * M
        QD[start_idx:end_idx, start_idx:end_idx] = Q_diagonal_sums[ii]
    
    QF = QD + QQ
    
    # Solve linear system
    X = np.linalg.lstsq(QF, PP, rcond=None)[0]
    c = X.reshape(N, M)
    
    # Vectorized image correction
    zzc = np.zeros_like(zz)
    
    # Precompute correction surfaces for all images
    corrections = np.tensordot(c, p, axes=([1], [0]))  # Shape: (N, H, W)
    
    for ii in range(N):
        img_data = zz[ii].data
        mask = zz[ii].mask
        
        # Apply correction
        corrected = img_data + corrections[ii]
        corrected = corrected * (~mask).astype(np.float32)
        zzc[ii] = np.ma.masked_array(corrected, mask)
    
    # Final averaging and Zernike removal
    ZZ = np.ma.mean(zzc, axis=0)
    ZZ = zern.removeZernike(ZZ, idr)
    
    return ZZ