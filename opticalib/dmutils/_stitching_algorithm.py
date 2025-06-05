import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
from opticalib.ground import zernike as zern
from opticalib import typings as t
import functools
import time
try:
    import cupy as cp
    print("[STITCHING] GPU acceleration available.")
except ImportError:
    cp = None
    print("[STITCHING] No GPU acceleration available. Using multi-core CPU computation.")


def timer(func: t.Callable[...,t.Any]) -> t.Callable[...,t.Any]:
    """Decorator to time the execution of a function."""
    @functools.wraps(func)
    def wrapper(*args: t.Any, **kwargs: dict[str, t.Any]) -> t.Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        h = (end_time - start_time)//3600
        m = (end_time - start_time)%3600//60
        s = (end_time - start_time)%60
        print(f"Execution time: {int(h):02d}:{int(m):02d}:{s:.2f} (h:m:s)")
        return result
    return wrapper


@timer
def map_stitching(
    image_vector: t.CubeData, 
    fullmask: t.ImageData, 
    zern2fit: list[int],
    mp_chunk_size: int = 64
) -> t.ImageData:
    """
    Stitching algorithm.

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
    print("Computing Zernike basis...", end="\r", flush=True)
    N = image_vector.shape[0]
    MM = fullmask.copy()
    M = len(zern2fit)
    base = np.ma.masked_array(np.full(MM.shape, 2, dtype=float), MM)
    _, mat = zern.zernikeFit(base, zern2fit)
    temp = np.tri(M, dtype=float)
    p = np.array([zern.zernikeSurface(base, temp[i], mat) for i in range(M)])
    Qo = np.tile(p, (M, 1, 1))
    v_order = np.reshape(np.reshape(np.arange(M**2), (M, M)).T, (1, M**2))
    q = Qo * Qo[v_order[0]]

    print("Setting up stitching algorithm...", end="\r", flush=True)
    # Pre-extract masks and data for efficiency
    masks = np.array([img.mask for img in image_vector])
    data = np.array([img.data for img in image_vector])

    # Prepare all (ii, jj) pairs
    pairs = [(ii, jj) for ii in range(N) for jj in range(N)]

    Q = np.zeros((N, N, M**2))
    P = np.zeros((N, N, M))

    if cp is not None:
        # GPU computation, if available
        # Move arrays to GPU
        masks_gpu = cp.asarray(masks)
        data_gpu = cp.asarray(data)
        p_gpu = cp.asarray(p)
        q_gpu = cp.asarray(q)

        for ii in trange(N, desc="P-Q Computation", ncols=80, colour="green"):
            for jj in range(N):
                mm = cp.logical_or(masks_gpu[ii], masks_gpu[jj])
                if ii == jj:
                    Q_val = cp.zeros(M**2, dtype=cp.float32)
                else:
                    Q_val = cp.sum(q_gpu * (~mm).astype(cp.float32), axis=(1, 2))
                img = data_gpu[ii] - data_gpu[jj]
                P_val = cp.nansum(p_gpu * (~mm).astype(cp.float32) * img, axis=(1, 2))
                Q[ii, jj, :] = cp.asnumpy(Q_val)
                P[ii, jj, :] = cp.asnumpy(P_val)
    else:
        # Back to CPU computation
        time.sleep(0.5)
        block_compute = _BlockCompute(masks, data, M, p, q)
        with Pool(processes=cpu_count()) as pool:
            for (ii, jj), Q_val, P_val in tqdm(
                pool.imap_unordered(block_compute.compute_block, pairs, chunksize=mp_chunk_size),
                desc='P-Q Computation',
                total=len(pairs),
                ncols=80,
                unit='pair',
                colour='green'
            ):
                Q[ii, jj, :] = Q_val
                P[ii, jj, :] = P_val

    print("Computing stitched image...", end="\r", flush=True)
    
    P1 = np.reshape(P, (N, N, M))
    Pt = [np.sum(P1[ii], axis=0) for ii in range(N)]
    PP = np.reshape(Pt, M * N)
    Q1 = np.reshape(Q, (N, N, M**2))
    QQ = np.reshape(Q1, (N, N, M, M))
    temp = np.vstack([np.hstack([QQ[ii, jj, :, :] for ii in range(N)]) for jj in range(N)])
    QQ = temp.copy()
    QD = np.zeros_like(QQ)
    for ii in range(N):
        temp = np.sum(Q1[ii], axis=0)
        temp = np.reshape(temp, (M, M))
        QD[M * ii : M * (ii + 1), M * ii : M * (ii + 1)] = -temp
    QF = QD + QQ
    X = np.linalg.lstsq(QF, PP, rcond=None)[0]
    zzc = np.zeros_like(image_vector)
    c = np.reshape(X, (N, M))
    for ii in range(N):
        img = image_vector[ii, :, :].data
        mm = image_vector[ii, :, :].mask
        res = np.zeros_like(MM)
        for ki in range(M):
            res += p[ki] * c[ii, ki]
        zzc[ii, :, :] = np.ma.masked_array((img + res) * (~mm).astype(float), mm)
    print(f"Removing zernike modes {zern2fit}...", end="\r", flush=True)
    ZZ = np.ma.mean(zzc, axis=0)
    ZZ = zern.removeZernike(ZZ, zern2fit)
    return ZZ


class _BlockCompute():

    def __init__(self, masks: t.ImageData, data: t.ImageData, M: int, p: t.ArrayLike, q: t.ArrayLike):
        self.masks = masks
        self.data = data
        self.M = M
        self.p = p
        self.q = q

    def compute_block(self, args: tuple[int, int]) -> tuple[tuple[int, int], t.MatrixLike, t.MatrixLike]:
        ii, jj = args
        mm = np.logical_or(self.masks[ii], self.masks[jj])
        if ii == jj:
            Q_val = np.zeros(self.M**2)
        else:
            Q_val = np.sum(self.q * (~mm).astype(float), axis=(1, 2))
        img = self.data[ii] - self.data[jj]
        P_val = np.array(np.nansum(self.p * (~mm).astype(float) * img, axis=(1, 2)))
        return (ii, jj), Q_val, P_val