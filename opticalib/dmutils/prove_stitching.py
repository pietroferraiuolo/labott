"""
Memory-optimized implementation of interferometric stitching algorithm.
Based on Otsubo et al. (1994) methodology with optimized linear algebra and memory management.
"""

import numpy as np
from typing import List, Tuple, Callable, Any
from opticalib.ground import zernike as zern
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
def map_stitching_fast(zz: np.ndarray, fullmask: np.ndarray, idr: List[int]) -> np.ndarray:
    """
    Memory-optimized interferometric stitching using vectorized operations.
    
    Parameters
    ----------
    zz : np.ndarray
        3D array of images to be stitched, shape (N, H, W).
    fullmask : np.ndarray
        2D array representing the full mask, shape (H, W).
    idr : List[int]
        List of Zernike indices to fit and remove.
        
    Returns
    -------
    np.ndarray
        Stitched and corrected surface map.
    """
    N, H, W = zz.shape
    M = len(idr)
    
    try:
        # Pre-compute Zernike basis functions once
        zernike_basis = _compute_zernike_basis(fullmask, idr)  # Shape: (M, H, W)
        
        # Process overlaps in chunks to avoid massive memory allocation
        Q_matrix, P_vector = _assemble_system_memory_efficient(zz, zernike_basis)
        
        # Clear intermediate data
        del zernike_basis
        gc.collect()
        
        # Solve linear system
        coefficients = np.linalg.solve(Q_matrix, P_vector)  # Shape: (N*M,)
        coefficients = coefficients.reshape(N, M)
        
        # Clear system matrices
        del Q_matrix, P_vector
        gc.collect()
        
        # Apply corrections efficiently
        final_map = _apply_corrections_memory_efficient(zz, fullmask, idr, coefficients)
        
        return final_map
        
    finally:
        # Ensure cleanup even on exceptions
        gc.collect()


def _compute_zernike_basis(mask: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Pre-compute all Zernike basis functions with memory optimization.
    
    Returns
    -------
    np.ndarray
        Zernike basis functions, shape (M, H, W).
    """
    M = len(indices)
    H, W = mask.shape
    
    # Create base for Zernike fitting
    base = np.ma.masked_array(np.ones((H, W), dtype=np.float32) * 2, mask)
    _, mat = zern.zernikeFit(base, indices)
    
    # Pre-allocate output with float32 to save memory
    basis = np.zeros((M, H, W), dtype=np.float32)
    
    # Compute each basis function
    temp_coeffs = np.zeros(M, dtype=np.float32)
    for i in range(M):
        temp_coeffs.fill(0)
        temp_coeffs[i] = 1
        basis[i] = zern.zernikeSurface(base, temp_coeffs, mat).astype(np.float32)
    
    # Clean up temporaries
    del base, mat, temp_coeffs
    gc.collect()
    
    return basis


def _assemble_system_memory_efficient(zz: np.ndarray, 
                                    zernike_basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient assembly of the linear system Q*x = P.
    Processes image pairs in chunks to avoid massive temporary arrays.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Q matrix (N*M, N*M) and P vector (N*M,).
    """
    N, H, W = zz.shape
    M = len(zernike_basis)
    
    # Pre-allocate matrices with float32 for memory efficiency
    Q_matrix = np.zeros((N * M, N * M), dtype=np.float32)
    P_vector = np.zeros(N * M, dtype=np.float32)
    
    # Process image pairs one at a time to avoid memory explosion
    for i in range(N):
        # Extract valid mask for image i once
        valid_i = ~zz[i].mask
        
        for j in range(N):
            if i == j:
                continue
            
            # Extract valid mask for image j
            valid_j = ~zz[j].mask
            
            # Compute overlap (small boolean array)
            overlap = valid_i & valid_j
            
            if not overlap.any():
                continue
            
            # Compute Q matrix block efficiently
            q_block = _compute_q_block_efficient(zernike_basis, overlap)
            Q_matrix[i*M:(i+1)*M, j*M:(j+1)*M] = q_block
            
            # Compute P vector block efficiently  
            img_diff = zz[i].data.astype(np.float32) - zz[j].data.astype(np.float32)
            p_block = _compute_p_block_efficient(zernike_basis, img_diff, overlap)
            P_vector[i*M:(i+1)*M] += p_block
            
            # Clean up temporaries
            del img_diff, overlap, valid_j
        
        del valid_i
    
    # Set diagonal blocks efficiently
    for i in range(N):
        row_sums = np.sum(Q_matrix[i*M:(i+1)*M, :], axis=1)
        Q_matrix[i*M:(i+1)*M, i*M:(i+1)*M] = -np.diag(row_sums)
    
    return Q_matrix, P_vector


def _compute_q_block_efficient(zernike_basis: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """
    Efficiently compute Q matrix block without creating huge intermediate arrays.
    """
    M = len(zernike_basis)
    q_block = np.zeros((M, M), dtype=np.float32)
    
    # Extract basis functions only in overlap region
    overlap_indices = np.where(overlap)
    if len(overlap_indices[0]) == 0:
        return q_block
    
    # Extract basis values at overlap points (much smaller arrays)
    basis_overlap = zernike_basis[:, overlap_indices[0], overlap_indices[1]]  # Shape: (M, n_overlap)
    
    # Compute outer product efficiently
    q_block = basis_overlap @ basis_overlap.T  # Shape: (M, M)
    
    return q_block


def _compute_p_block_efficient(zernike_basis: np.ndarray, 
                             img_diff: np.ndarray, 
                             overlap: np.ndarray) -> np.ndarray:
    """
    Efficiently compute P vector block.
    """
    M = len(zernike_basis)
    
    # Extract only overlap region data
    overlap_indices = np.where(overlap)
    if len(overlap_indices[0]) == 0:
        return np.zeros(M, dtype=np.float32)
    
    # Extract basis and image difference at overlap points
    basis_overlap = zernike_basis[:, overlap_indices[0], overlap_indices[1]]  # Shape: (M, n_overlap)
    img_diff_overlap = img_diff[overlap_indices[0], overlap_indices[1]]      # Shape: (n_overlap,)
    
    # Vectorized inner product
    p_block = basis_overlap @ img_diff_overlap  # Shape: (M,)
    
    return p_block.astype(np.float32)


def _apply_corrections_memory_efficient(zz: np.ndarray, 
                                      fullmask: np.ndarray,
                                      idr: List[int],
                                      coefficients: np.ndarray) -> np.ndarray:
    """
    Apply Zernike corrections efficiently without storing all basis functions.
    """
    N, H, W = zz.shape
    M = len(idr)
    
    # Create accumulator for final result
    result_sum = np.ma.zeros((H, W), dtype=np.float64)
    result_count = np.zeros((H, W), dtype=np.int32)
    
    # Create base for Zernike operations (reuse for all images)
    base = np.ma.masked_array(np.ones((H, W), dtype=np.float32) * 2, fullmask)
    _, mat = zern.zernikeFit(base, idr)
    
    # Process each image individually to save memory
    for i in range(N):
        # Compute correction for this image
        correction = zern.zernikeSurface(base, coefficients[i].astype(np.float32), mat)
        
        # Apply correction
        corrected_img = zz[i].data.astype(np.float64) + correction.astype(np.float64)
        
        # Accumulate valid pixels only
        valid_mask = ~zz[i].mask
        result_sum.data[valid_mask] += corrected_img[valid_mask]
        result_count[valid_mask] += 1
        
        # Update mask
        result_sum.mask &= zz[i].mask
        
        # Clean up temporaries
        del correction, corrected_img, valid_mask
        gc.collect()
    
    # Compute average where we have data
    valid_average = result_count > 0
    final_result = np.ma.zeros_like(result_sum)
    final_result.data[valid_average] = result_sum.data[valid_average] / result_count[valid_average]
    final_result.mask = ~valid_average | result_sum.mask
    
    # Remove specified Zernikes
    final_result = zern.removeZernike(final_result, idr)
    
    # Final cleanup
    del result_sum, result_count, base, mat
    gc.collect()
    
    return final_result


# Chunked processing version for extremely large datasets
@timer
def map_stitching_chunked(zz: np.ndarray, fullmask: np.ndarray, idr: List[int], 
                         chunk_size: int = 5) -> np.ndarray:
    """
    Process very large datasets in chunks to avoid memory overflow.
    
    Parameters
    ----------
    chunk_size : int
        Number of images to process simultaneously.
    """
    N, H, W = zz.shape
    M = len(idr)
    
    if N <= chunk_size:
        return map_stitching_fast(zz, fullmask, idr)
    
    # Process in chunks and accumulate results
    result_accumulator = np.ma.zeros((H, W), dtype=np.float64)
    weight_accumulator = np.zeros((H, W), dtype=np.float64)
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = zz[start_idx:end_idx]
        
        # Process chunk
        chunk_result = map_stitching_fast(chunk, fullmask, idr)
        
        # Accumulate with weights
        valid_pixels = ~chunk_result.mask
        result_accumulator.data[valid_pixels] += chunk_result.data[valid_pixels]
        weight_accumulator[valid_pixels] += 1.0
        
        # Update mask
        result_accumulator.mask &= chunk_result.mask
        
        # Clean up
        del chunk, chunk_result
        gc.collect()
    
    # Final averaging
    valid_avg = weight_accumulator > 0
    final_result = np.ma.zeros_like(result_accumulator)
    final_result.data[valid_avg] = result_accumulator.data[valid_avg] / weight_accumulator[valid_avg]
    final_result.mask = ~valid_avg | result_accumulator.mask
    
    return final_result