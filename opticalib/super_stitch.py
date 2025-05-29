import numpy as np
from typing import List, Optional, Tuple
from opticalib.ground import zernike as zern
from numba import jit, prange
import concurrent.futures
from functools import lru_cache


@jit(nopython=True, parallel=True)
def _compute_q_elements_numba(
    p_ki: np.ndarray, 
    p_kj: np.ndarray, 
    valid_pixels: np.ndarray
) -> float:
    """Fast computation of Q matrix elements using Numba."""
    return np.sum(p_ki * p_kj * valid_pixels)


@jit(nopython=True, parallel=True)
def _compute_p_elements_numba(
    p_ki: np.ndarray, 
    valid_pixels: np.ndarray, 
    img_diff: np.ndarray
) -> float:
    """Fast computation of P vector elements using Numba."""
    return np.sum(p_ki * valid_pixels * img_diff)


@jit(nopython=True, parallel=True)
def _apply_correction_numba(
    img_data: np.ndarray,
    correction: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Fast correction application using Numba."""
    result = img_data + correction
    for i in prange(result.shape[0]):
        for j in prange(result.shape[1]):
            if mask[i, j]:
                result[i, j] = 0.0
    return result


def _process_image_pair_worker(args: Tuple) -> Tuple:
    """
    Worker function for processing image pairs (must be module-level for pickling).
    
    Parameters
    ----------
    args : Tuple
        Contains (ii, jj, zz_ii_data, zz_ii_mask, zz_jj_data, zz_jj_mask, p_surfaces, M)
        
    Returns
    -------
    Tuple
        (ii, jj, q_elements, p_elements)
    """
    ii, jj, zz_ii_data, zz_ii_mask, zz_jj_data, zz_jj_mask, p_surfaces, M = args
    
    if ii == jj:
        return ii, jj, None, None
        
    # Compute mask
    mask = np.logical_or(zz_ii_mask, zz_jj_mask)
    valid_pixels = (~mask).astype(np.float32)
    
    # Compute Q elements (exploit symmetry)
    q_elements = np.zeros((M, M), dtype=np.float32)
    for ki in range(M):
        for kj in range(ki, M):  # Only upper triangle
            q_val = _compute_q_elements_numba(
                p_surfaces[ki], p_surfaces[kj], valid_pixels
            )
            q_elements[ki, kj] = q_val
            if ki != kj:
                q_elements[kj, ki] = q_val
    
    # Compute P elements
    img_diff = zz_ii_data.astype(np.float32) - zz_jj_data.astype(np.float32)
    p_elements = np.zeros(M, dtype=np.float64)
    
    for ki in range(M):
        p_elements[ki] = _compute_p_elements_numba(
            p_surfaces[ki], valid_pixels, img_diff
        )
    
    return ii, jj, q_elements, p_elements


def _apply_correction_worker(args: Tuple) -> np.ndarray:
    """
    Worker function for applying corrections (must be module-level for pickling).
    
    Parameters
    ----------
    args : Tuple
        Contains (ii, img_data, mask, correction_coeffs, p_surfaces, H, W, M)
        
    Returns
    -------
    np.ndarray
        Corrected image
    """
    ii, img_data, mask, correction_coeffs, p_surfaces, H, W, M = args
    
    # Compute correction
    correction = np.zeros((H, W), dtype=np.float32)
    for ki in range(M):
        correction += p_surfaces[ki] * correction_coeffs[ki]
    
    # Apply correction using Numba
    corrected = _apply_correction_numba(
        img_data.astype(np.float32), correction, mask
    )
    
    return corrected


class ZernikeCache:
    """Efficient caching for Zernike surfaces."""
    
    def __init__(self, base: np.ndarray, mat: np.ndarray, M: int):
        self.base = base
        self.mat = mat
        self.M = M
        self._cache = {}
    
    def get_surface(self, idx: int) -> np.ndarray:
        """Get Zernike surface with caching."""
        if idx not in self._cache:
            temp = np.zeros(self.M, dtype=np.float32)
            temp[idx] = 1
            surface = zern.zernikeSurface(self.base, temp, self.mat)
            self._cache[idx] = surface.astype(np.float32)
        return self._cache[idx]


def map_stitching(
    zz: np.ndarray, 
    fullmask: np.ndarray, 
    idr: List[int],
    chunk_size: Optional[int] = None,
    n_workers: Optional[int] = None,
    use_sparse: bool = True
) -> np.ndarray:
    """
    Ultra-fast memory-optimized stitching using parallel processing and Numba.
    
    Parameters
    ----------
    zz : np.ndarray
        3D array of images to be stitched, shape (N, H, W).
    fullmask : np.ndarray
        2D array representing the full mask, shape (H, W).
    idr : List[int]
        List of Zernike indices to fit.
    chunk_size : Optional[int]
        Chunk size for processing. Auto-determined if None.
    n_workers : Optional[int]
        Number of parallel workers. Auto-determined if None.
    use_sparse : bool
        Whether to use sparse matrix operations for large systems.
        
    Returns
    -------
    np.ndarray
        Stitched and corrected image.
    """
    N = zz.shape[0]
    M = len(idr)
    H, W = zz.shape[1], zz.shape[2]
    
    # Auto-configure parameters
    if chunk_size is None:
        memory_factor = min(8, max(1, 32 // N))
        chunk_size = max(1, N // memory_factor)
    
    if n_workers is None:
        n_workers = min(4, max(1, N // 4))  # Reduced for better stability
    
    # Precompute Zernike basis
    base_array = np.ones((H, W), dtype=np.float32) * 2
    base = np.ma.masked_array(base_array, fullmask)
    _, mat = zern.zernikeFit(base, idr)
    
    # Initialize Zernike cache
    zernike_cache = ZernikeCache(base, mat, M)
    
    # Precompute all Zernike surfaces
    p_surfaces = []
    for i in range(M):
        p_surfaces.append(zernike_cache.get_surface(i))
    
    # Convert to numpy array for easier serialization
    p_surfaces = np.array(p_surfaces)
    
    # Use sparse matrices for large systems
    if use_sparse and N * M > 1000:
        from scipy import sparse
        QQ = sparse.lil_matrix((N * M, N * M), dtype=np.float32)
        use_sparse_solver = True
    else:
        QQ = np.zeros((N * M, N * M), dtype=np.float32)
        use_sparse_solver = False
    
    PP = np.zeros(N * M, dtype=np.float64)
    
    # Process in chunks with parallel execution
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        
        # Prepare arguments for parallel processing
        args_list = []
        for ii in range(i_start, i_end):
            for jj in range(N):
                args_list.append((
                    ii, jj, 
                    zz[ii].data, zz[ii].mask,
                    zz[jj].data, zz[jj].mask,
                    p_surfaces, M
                ))
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor for better compatibility
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_image_pair_worker, args_list))
        
        # Accumulate results
        for ii, jj, q_elements, p_elements in results:
            if q_elements is not None:
                # Fill Q matrix
                i_slice = slice(ii * M, (ii + 1) * M)
                j_slice = slice(jj * M, (jj + 1) * M)
                
                if use_sparse_solver:
                    QQ[i_slice, j_slice] = q_elements
                else:
                    QQ[i_slice, j_slice] = q_elements
                
                # Accumulate P vector (sum over j for each i)
                PP[i_slice] += p_elements
    
    # Add diagonal terms to Q matrix
    for ii in range(N):
        i_slice = slice(ii * M, (ii + 1) * M)
        
        if use_sparse_solver:
            diagonal_sum = -np.sum([QQ[i_slice, jj * M:(jj + 1) * M].toarray() 
                                   for jj in range(N) if jj != ii], axis=0)
            QQ[i_slice, i_slice] = diagonal_sum
        else:
            diagonal_sum = -np.sum([QQ[i_slice, jj * M:(jj + 1) * M] 
                                   for jj in range(N) if jj != ii], axis=0)
            QQ[i_slice, i_slice] = diagonal_sum
    
    # Solve linear system
    if use_sparse_solver:
        from scipy.sparse.linalg import spsolve
        QQ = QQ.tocsr()
        X = spsolve(QQ, PP)
    else:
        X = np.linalg.lstsq(QQ, PP, rcond=None)[0]
    
    c = X.reshape(N, M)
    
    # Prepare arguments for parallel correction
    correction_args = [
        (ii, zz[ii].data, zz[ii].mask, c[ii], p_surfaces, H, W, M) 
        for ii in range(N)
    ]
    
    # Apply corrections in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        corrected_images = list(executor.map(_apply_correction_worker, correction_args))
    
    # Final averaging with proper masking
    total_sum = np.zeros((H, W), dtype=np.float64)
    total_count = np.zeros((H, W), dtype=np.int32)
    
    for ii, corrected in enumerate(corrected_images):
        valid_mask = ~zz[ii].mask
        total_sum[valid_mask] += corrected[valid_mask].astype(np.float64)
        total_count[valid_mask] += 1
    
    # Compute final result
    final_mask = total_count == 0
    ZZ_data = np.zeros((H, W), dtype=np.float32)
    ZZ_data[~final_mask] = (total_sum[~final_mask] / total_count[~final_mask]).astype(np.float32)
    
    ZZ = np.ma.masked_array(ZZ_data, final_mask)
    ZZ = zern.removeZernike(ZZ, idr)
    
    return ZZ