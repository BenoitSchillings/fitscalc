import sys
import os
import numpy as np
import cv2
from astropy.io import fits
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer

import pyqtgraph as pg
import argparse

from scipy.optimize import leastsq


from scipy.optimize import curve_fit


#!/usr/bin/env python3
"""
Ultra-fast pixel outlier filter for 2D numpy arrays.

This module provides an optimized outlier detection and filtering function
that can process astronomical images at rates exceeding 250 MPixels/sec.
"""

import numpy as np
from numba import jit, prange
import numba

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def filter_outliers(image, sigma_threshold=3.0, in_place=True):
    """
    Ultra-fast outlier detection and filtering for 2D images.

    Detects pixels that deviate significantly from their local 5×5 neighborhood
    and replaces them with the local mean. Optimized for maximum performance
    while maintaining high accuracy.

    Parameters:
    -----------
    image : numpy.ndarray
        2D image array to filter (will be modified if in_place=True)
    sigma_threshold : float, optional
        Number of standard deviations for outlier threshold (default: 3.0)
        Higher values = more conservative (fewer corrections)
        Recommended: 3.0-4.5 depending on noise level
    in_place : bool, optional
        If True, modifies the input array directly (default: True)
        If False, returns a copy of the filtered image

    Returns:
    --------
    filtered_image : numpy.ndarray
        The filtered image (same as input if in_place=True)
    corrections : int
        Number of pixels that were corrected

    Performance:
    ------------
    - Processing rate: >250 MPixels/sec on modern CPUs
    - Memory efficient: minimal additional allocation
    - Parallel processing: utilizes multiple CPU cores

    Examples:
    ---------
    >>> import numpy as np
    >>> from pixel_filter import filter_outliers
    >>>
    >>> # Create test image with noise and outliers
    >>> image = np.random.normal(1000, 10, (2048, 2048))
    >>> image[100, 100] = 2000  # Add outlier
    >>>
    >>> # Filter outliers
    >>> filtered_image, corrections = filter_outliers(image)
    >>> print(f"Corrected {corrections} pixels")
    >>>
    >>> # Use custom threshold
    >>> filtered_image, corrections = filter_outliers(image, sigma_threshold=4.0)
    """
    h, w = image.shape

    # Work on copy if not in-place
    if not in_place:
        image = image.copy()

    # Pre-compute constants for performance
    sigma_threshold_sq = sigma_threshold * sigma_threshold
    inv_24 = 1.0 / 24.0   # 5×5 kernel has 24 neighbors (excluding center)
    inv_23 = 1.0 / 23.0   # Sample variance uses n-1 denominator
    min_variance = 1e-10  # Avoid division by near-zero variance

    corrections = 0

    # Process image with 2-pixel border (required for 5×5 kernel)
    for y in prange(2, h - 2):
        for x in range(2, w - 2):
            center_value = image[y, x]

            # Calculate sum of 24 neighboring pixels (5×5 excluding center)
            neighbor_sum = (
                # Top two rows
                image[y-2, x-2] + image[y-2, x-1] + image[y-2, x] + image[y-2, x+1] + image[y-2, x+2] +
                image[y-1, x-2] + image[y-1, x-1] + image[y-1, x] + image[y-1, x+1] + image[y-1, x+2] +
                # Middle row (excluding center)
                image[y, x-2] + image[y, x-1] + image[y, x+1] + image[y, x+2] +
                # Bottom two rows
                image[y+1, x-2] + image[y+1, x-1] + image[y+1, x] + image[y+1, x+1] + image[y+1, x+2] +
                image[y+2, x-2] + image[y+2, x-1] + image[y+2, x] + image[y+2, x+1] + image[y+2, x+2]
            )

            # Calculate neighborhood mean
            neighbor_mean = neighbor_sum * inv_24

            # Calculate sample variance of neighborhood
            sum_squared_deviations = (
                # Top two rows
                (image[y-2, x-2] - neighbor_mean)**2 + (image[y-2, x-1] - neighbor_mean)**2 +
                (image[y-2, x] - neighbor_mean)**2 + (image[y-2, x+1] - neighbor_mean)**2 +
                (image[y-2, x+2] - neighbor_mean)**2 +
                (image[y-1, x-2] - neighbor_mean)**2 + (image[y-1, x-1] - neighbor_mean)**2 +
                (image[y-1, x] - neighbor_mean)**2 + (image[y-1, x+1] - neighbor_mean)**2 +
                (image[y-1, x+2] - neighbor_mean)**2 +
                # Middle row (excluding center)
                (image[y, x-2] - neighbor_mean)**2 + (image[y, x-1] - neighbor_mean)**2 +
                (image[y, x+1] - neighbor_mean)**2 + (image[y, x+2] - neighbor_mean)**2 +
                # Bottom two rows
                (image[y+1, x-2] - neighbor_mean)**2 + (image[y+1, x-1] - neighbor_mean)**2 +
                (image[y+1, x] - neighbor_mean)**2 + (image[y+1, x+1] - neighbor_mean)**2 +
                (image[y+1, x+2] - neighbor_mean)**2 +
                (image[y+2, x-2] - neighbor_mean)**2 + (image[y+2, x-1] - neighbor_mean)**2 +
                (image[y+2, x] - neighbor_mean)**2 + (image[y+2, x+1] - neighbor_mean)**2 +
                (image[y+2, x+2] - neighbor_mean)**2
            )

            # Sample variance (using n-1 denominator)
            variance = sum_squared_deviations * inv_23

            # Check if center pixel is an outlier
            if variance > min_variance:
                center_deviation = center_value - neighbor_mean
                deviation_squared_normalized = (center_deviation * center_deviation) / variance

                # If pixel deviates more than threshold, replace with neighborhood mean
                if deviation_squared_normalized > sigma_threshold_sq:
                    image[y, x] = neighbor_mean
                    corrections += 1

    return image, corrections


def filter_outliers_simple(image, sigma_threshold=3.0, in_place=True):
    """
    Simple wrapper function that only returns the filtered image.

    Parameters:
    -----------
    image : numpy.ndarray
        2D image array to filter
    sigma_threshold : float, optional
        Number of standard deviations for outlier threshold (default: 3.0)
    in_place : bool, optional
        If True, modifies the input array directly (default: True)

    Returns:
    --------
    filtered_image : numpy.ndarray
        The filtered image
    """
    filtered_image, _ = filter_outliers(image, sigma_threshold, in_place)
    return filtered_image


if __name__ == "__main__":
    # Simple performance test
    import time

    print("🚀 Pixel Filter Performance Test")
    print("=" * 40)

    # Create test image
    size = 2048
    test_image = np.random.normal(1000, 10, (size, size)).astype(np.float64)

    # Add some outliers
    for _ in range(100):
        y, x = np.random.randint(10, size-10, 2)
        test_image[y, x] = 1000 + np.random.choice([-1, 1]) * 100

    print(f"Test image: {size}×{size} ({test_image.nbytes/(1024**2):.1f} MB)")

    # Warm up JIT compilation
    _ = filter_outliers(test_image[:100, :100].copy())

    # Benchmark
    start_time = time.perf_counter()
    filtered_image, corrections = filter_outliers(test_image.copy())
    elapsed_time = time.perf_counter() - start_time

    rate = (size * size) / elapsed_time / 1e6

    print(f"Processing time: {elapsed_time*1000:.2f} ms")
    print(f"Processing rate: {rate:.1f} MPixels/sec")
    print(f"Outliers corrected: {corrections}")
    print(f"Correction rate: {100*corrections/(size*size):.3f}%")
    print("\n✅ Ready for production use!")
    


class BayesianPhotonCounter:
    """
    Fast Bayesian photon counting for EMCCDs using lookup table.

    Eliminates the sqrt(2) excess noise factor from stochastic EM gain
    by estimating the most likely photon count per pixel using Bayesian
    inference, precomputed into a lookup table for real-time performance.

    Based on: Harpsøe et al. (2012) "Bayesian photon counting with EMCCDs"
    A&A 537, A50. https://doi.org/10.1051/0004-6361/201117089

    Performance: ~300 Mpix/sec (3ms for 1024x1024)

    Parameters
    ----------
    em_gain : float
        EM gain setting (typically 100-1000)
    read_noise : float
        Read noise in electrons (typically 3-50 for EMCCDs)
    bias : float
        Bias level to subtract (default 0)
    max_adu : int
        Maximum ADU value to precompute (default 20000)
    max_photons : int
        Maximum photon count to consider in model (default 6)
    lambda_resolution : int
        Number of lambda values to test (default 100)
    lambda_max : float
        Maximum lambda (photons/pixel) to estimate (default 5.0)

    Example
    -------
    >>> counter = BayesianPhotonCounter(em_gain=300, read_noise=3.0)
    >>> # Single frame
    >>> photon_map = counter.estimate(frame)
    >>> # Accumulate multiple frames
    >>> counter.reset_accumulator(frame.shape)
    >>> for frame in frames:
    ...     counter.accumulate(frame)
    >>> total_photons = counter.get_accumulated()
    """

    def __init__(self, em_gain, read_noise, bias=0, max_adu=20000,
                 max_photons=6, lambda_resolution=100, lambda_max=5.0):
        from scipy.stats import poisson, gamma as gamma_dist
        from scipy.ndimage import gaussian_filter1d

        self.em_gain = em_gain
        self.read_noise = read_noise
        self.bias = bias
        self.max_adu = max_adu

        # Build lookup table using accurate gamma distributions
        lambdas = np.linspace(0.001, lambda_max, lambda_resolution)
        adu_values = np.arange(max_adu, dtype=np.float64)

        # Pre-compute gamma PDFs for each photon count, convolved with read noise
        # This is more accurate than Gaussian approximation for low k
        gamma_pdfs = {}
        for k in range(1, max_photons + 1):
            # Gamma(k, scale=gain) PDF
            # For k=1, this is exponential distribution
            pdf = gamma_dist.pdf(adu_values, a=k, scale=em_gain)
            # Convolve with Gaussian read noise
            if read_noise > 0.5:
                pdf = gaussian_filter1d(pdf, sigma=read_noise)
            # Normalize
            pdf = pdf / (pdf.sum() + 1e-30)
            gamma_pdfs[k] = pdf

        # Zero photon case: Gaussian centered at 0
        zero_pdf = np.exp(-0.5 * (adu_values / max(read_noise, 0.5))**2)
        zero_pdf = zero_pdf / (zero_pdf.sum() + 1e-30)

        # Likelihood table: P(adu | lambda) for each adu and lambda
        likelihoods = np.zeros((max_adu, lambda_resolution))

        for i, lam in enumerate(lambdas):
            prob = np.zeros(max_adu)
            for k in range(max_photons + 1):
                # P(k photons | lambda) - Poisson
                p_k = poisson.pmf(k, lam)

                if k == 0:
                    p_adu_given_k = zero_pdf
                else:
                    p_adu_given_k = gamma_pdfs[k]

                prob += p_k * p_adu_given_k

            likelihoods[:, i] = prob

        # Store best lambda for each ADU value
        best_idx = np.argmax(likelihoods, axis=1)
        self.lut = lambdas[best_idx].astype(np.float32)

        # Accumulator for multi-frame photon counting
        self._accumulator = None
        self._frame_count = 0

    def estimate(self, frame):
        """
        Estimate photon counts for a single frame.

        Parameters
        ----------
        frame : ndarray
            2D image array (uint16 or float)

        Returns
        -------
        photon_map : ndarray
            Estimated photon count per pixel (float32)
        """
        corrected = frame.astype(np.float32) - self.bias

        # Use LUT for values within range
        in_range = corrected < self.max_adu
        result = np.zeros_like(corrected)

        # LUT lookup for low values (photon counting regime)
        idx = np.clip(corrected, 0, self.max_adu - 1).astype(np.int32)
        result[in_range] = self.lut[idx[in_range]]

        # For high values (outside photon counting regime), use linear estimate:
        # At high flux, mean ADU ≈ n_photons * em_gain, so n ≈ ADU / gain
        # This is less accurate but better than clipping
        high_mask = ~in_range
        if high_mask.any():
            result[high_mask] = corrected[high_mask] / self.em_gain

        return result

    def reset_accumulator(self, shape):
        """Reset the accumulator for multi-frame photon counting."""
        self._accumulator = np.zeros(shape, dtype=np.float32)
        self._frame_count = 0

    def accumulate(self, frame):
        """
        Add a frame to the photon count accumulator.

        Parameters
        ----------
        frame : ndarray
            2D image array
        """
        if self._accumulator is None:
            self.reset_accumulator(frame.shape)

        self._accumulator += self.estimate(frame)
        self._frame_count += 1

    def get_accumulated(self):
        """
        Get the accumulated photon count image.

        Returns
        -------
        total : ndarray
            Total estimated photons per pixel
        """
        if self._accumulator is None:
            return None
        return self._accumulator.copy()

    @property
    def frame_count(self):
        """Number of frames accumulated."""
        return self._frame_count


class ThresholdPhotonCounter:
    """
    Simple threshold-based photon counting for EMCCDs.

    Faster than Bayesian but only works well at very low flux
    (< 0.5 photons/pixel/frame). Uses 5-sigma thresholding with
    coincidence loss correction.

    Parameters
    ----------
    em_gain : float
        EM gain setting
    read_noise : float
        Read noise in electrons
    bias : float
        Bias level to subtract
    threshold : float, optional
        Detection threshold. If None, uses max(5*read_noise, 0.2*em_gain)
    """

    def __init__(self, em_gain, read_noise, bias=0, threshold=None):
        self.em_gain = em_gain
        self.read_noise = read_noise
        self.bias = bias

        # Default threshold: 5σ read noise or 0.2× gain (whichever is higher)
        if threshold is None:
            self.threshold = max(5 * read_noise, 0.2 * em_gain)
        else:
            self.threshold = threshold

        self._accumulator = None
        self._frame_count = 0

    def detect(self, frame):
        """
        Detect photon events in a single frame (binary).

        Returns
        -------
        binary : ndarray
            1 where photon detected, 0 otherwise
        """
        corrected = frame.astype(np.float32) - self.bias
        return (corrected > self.threshold).astype(np.float32)

    def estimate(self, frame):
        """
        Estimate photons per pixel (same interface as BayesianPhotonCounter).

        For threshold counting, this returns binary detection (0 or 1).
        This also accumulates internally for coincidence-corrected totals.

        Returns
        -------
        photons : ndarray (float32)
            Binary detection (0 or 1) per pixel
        """
        detected = self.detect(frame)
        # Also accumulate for running totals
        if self._accumulator is None:
            self.reset_accumulator(frame.shape)
        self._accumulator += detected
        self._frame_count += 1
        return detected

    def reset_accumulator(self, shape):
        """Reset the accumulator."""
        self._accumulator = np.zeros(shape, dtype=np.float32)
        self._frame_count = 0

    def accumulate(self, frame):
        """Add a frame to the accumulator."""
        if self._accumulator is None:
            self.reset_accumulator(frame.shape)

        self._accumulator += self.detect(frame)
        self._frame_count += 1

    def get_accumulated(self, corrected=True, flat=None):
        """
        Get accumulated photon counts with optional coincidence correction.

        Parameters
        ----------
        corrected : bool
            Apply coincidence loss correction: λ = -N·ln(1 - k/N)
        flat : ndarray, optional
            Normalized flat field to divide by (applied after coincidence correction)

        Returns
        -------
        photons : ndarray
            Estimated total photons per pixel
        """
        if self._accumulator is None:
            return None

        if corrected and self._frame_count > 0:
            k = self._accumulator.astype(np.float64)
            N = self._frame_count
            # Avoid log(0) by clipping
            k = np.clip(k, 0, N - 0.001)
            result = (-N * np.log(1.0 - k / N)).astype(np.float32)
        else:
            result = self._accumulator.copy()

        # Apply flat field correction to accumulated counts
        if flat is not None:
            result = result / flat

        return result

    @property
    def frame_count(self):
        """Number of frames accumulated."""
        return self._frame_count


class MultiThresholdPhotonCounter:
    """
    Multi-threshold photon counting for EMCCDs.

    Extends photon counting to higher flux levels (up to ~2-3 photons/pixel)
    by using multiple thresholds to distinguish 0, 1, 2, 3+ photon events.

    Based on: Basden et al. (2003) and arxiv:2312.04184

    Parameters
    ----------
    em_gain : float
        EM gain setting
    read_noise : float
        Read noise in electrons
    bias : float
        Bias level to subtract
    max_photons : int
        Maximum distinguishable photon count (default 4)
    """

    def __init__(self, em_gain, read_noise, bias=0, max_photons=4):
        self.em_gain = em_gain
        self.read_noise = read_noise
        self.bias = bias
        self.max_photons = max_photons

        # Compute threshold boundaries between k and k+1 photons
        # These are approximate crossings of gamma(k,g) and gamma(k+1,g) PDFs
        self.thresholds = []
        for k in range(max_photons):
            if k == 0:
                # 0 vs 1: threshold at 5σ or 0.2×gain
                t = max(5 * read_noise, 0.2 * em_gain)
            else:
                # k vs k+1: crossing point approximation
                t = k * em_gain * np.log((k + 1) / k) + em_gain * 0.5
            self.thresholds.append(t)

        self._accumulator = None
        self._frame_count = 0

    def count(self, frame):
        """
        Count photons in a single frame using multi-thresholding.

        Returns
        -------
        counts : ndarray
            Estimated photon count per pixel (0, 1, 2, ... max_photons)
        """
        x = frame.astype(np.float32) - self.bias
        result = np.zeros_like(x)

        for k, thresh in enumerate(self.thresholds):
            result[x >= thresh] = k + 1

        return result

    def reset_accumulator(self, shape):
        """Reset the accumulator."""
        self._accumulator = np.zeros(shape, dtype=np.float32)
        self._frame_count = 0

    def accumulate(self, frame):
        """Add a frame to the accumulator."""
        if self._accumulator is None:
            self.reset_accumulator(frame.shape)

        self._accumulator += self.count(frame)
        self._frame_count += 1

    def get_accumulated(self):
        """Get accumulated photon counts."""
        if self._accumulator is None:
            return None
        return self._accumulator.copy()

    @property
    def frame_count(self):
        """Number of frames accumulated."""
        return self._frame_count


class MultiFrameBayesianCounter:
    """
    Multi-frame Bayesian photon counting for EMCCDs.

    This implements the method from Harpsøe et al. (2012) "Bayesian photon
    counting with EMCCDs", A&A 537, A50.

    Key insight: Single-frame photon estimation is fundamentally limited by
    the ambiguity between low-signal photon events and noise. By combining
    likelihoods across multiple frames, we can build a posterior distribution
    that achieves near shot-noise limited performance.

    P(λ|{n₁, n₂, ...}) ∝ ∏ᵢ P(nᵢ|λ) × P(λ)

    This method achieves ~0.90 of the shot-noise limit SNR, compared to
    ~0.72 for raw EMCCD integration (a ~1.26x improvement, close to the
    theoretical √2 = 1.41x maximum).

    Parameters
    ----------
    em_gain : float
        EM gain setting (typically 100-1000)
    read_noise : float
        Read noise in electrons
    bias : float
        Bias level to subtract from frames
    max_adu : int
        Maximum ADU value to consider (default 5000, lower = faster)
    max_photons : int
        Maximum photon count in likelihood model (default 6)
    lambda_resolution : int
        Number of λ values to test (default 200)
    lambda_max : float
        Maximum λ to estimate (default 5.0)

    Example
    -------
    >>> counter = MultiFrameBayesianCounter(em_gain=300, read_noise=3.0, bias=100)
    >>> # Add frames one at a time
    >>> for frame in frames:
    ...     counter.add_frame(frame)
    >>> # Get photon estimate (combines all frames)
    >>> photon_map = counter.get_estimate()
    >>> print(f"Processed {counter.frame_count} frames")

    Notes
    -----
    - Requires multiple frames of the same scene for accurate estimation
    - Works best for λ < 3 photons/pixel/frame
    - Memory usage scales with image size × lambda_resolution
    - For real-time single-frame display, use BayesianPhotonCounter instead
      (though it won't achieve the SNR improvement)
    """

    def __init__(self, em_gain, read_noise, bias=0, max_adu=5000,
                 max_photons=6, lambda_resolution=200, lambda_max=5.0):
        from scipy.stats import poisson, gamma as gamma_dist
        from scipy.ndimage import gaussian_filter1d

        self.em_gain = em_gain
        self.read_noise = read_noise
        self.bias = bias
        self.max_adu = max_adu

        # Lambda values to test
        self.lambdas = np.linspace(0.001, lambda_max, lambda_resolution)
        self.lambda_resolution = lambda_resolution
        adu_values = np.arange(max_adu, dtype=np.float64)

        # Pre-compute gamma PDFs for each photon count, convolved with read noise
        gamma_pdfs = {}
        for k in range(1, max_photons + 1):
            pdf = gamma_dist.pdf(adu_values, a=k, scale=em_gain)
            if read_noise > 0.5:
                pdf = gaussian_filter1d(pdf, sigma=read_noise)
            pdf = pdf / (pdf.sum() + 1e-30)
            gamma_pdfs[k] = pdf

        # Zero photon case: Gaussian centered at 0
        zero_pdf = np.exp(-0.5 * (adu_values / max(read_noise, 0.5))**2)
        zero_pdf = zero_pdf / (zero_pdf.sum() + 1e-30)

        # Build log-likelihood table: log P(adu | λ)
        # Shape: (max_adu, lambda_resolution)
        self.log_likelihood = np.zeros((max_adu, lambda_resolution))

        for i, lam in enumerate(self.lambdas):
            prob = np.zeros(max_adu)
            for k in range(max_photons + 1):
                p_k = poisson.pmf(k, lam)
                if k == 0:
                    prob += p_k * zero_pdf
                else:
                    prob += p_k * gamma_pdfs[k]
            self.log_likelihood[:, i] = np.log(prob + 1e-300)

        # Accumulator for log-posterior
        self._log_posterior = None
        self._frame_count = 0
        self._shape = None

    def reset(self, shape=None):
        """Reset the accumulator for a new measurement."""
        if shape is not None:
            self._shape = shape
        if self._shape is not None:
            self._log_posterior = np.zeros((*self._shape, self.lambda_resolution),
                                           dtype=np.float32)
        self._frame_count = 0

    def add_frame(self, frame):
        """
        Add a frame to the Bayesian accumulator.

        Each frame's log-likelihood is added to the running log-posterior.

        Parameters
        ----------
        frame : ndarray
            2D image array (uint16 or float)
        """
        if self._log_posterior is None:
            self.reset(frame.shape)

        # Subtract bias and clip to valid range
        corrected = frame.astype(np.float32) - self.bias
        adu_idx = np.clip(corrected, 0, self.max_adu - 1).astype(np.int32)

        # Add log-likelihood from this frame to log-posterior
        self._log_posterior += self.log_likelihood[adu_idx]
        self._frame_count += 1

    def get_estimate(self):
        """
        Get the MAP (maximum a posteriori) estimate of λ per pixel.

        Returns
        -------
        lambda_map : ndarray
            Estimated photon rate per pixel (photons/frame)
        """
        if self._log_posterior is None or self._frame_count == 0:
            return None

        # Find λ that maximizes posterior for each pixel
        best_idx = np.argmax(self._log_posterior, axis=-1)
        return self.lambdas[best_idx]

    def get_total_photons(self):
        """
        Get estimated total photons per pixel (λ × n_frames).

        Returns
        -------
        total : ndarray
            Estimated total photons per pixel across all frames
        """
        estimate = self.get_estimate()
        if estimate is None:
            return None
        return estimate * self._frame_count

    @property
    def frame_count(self):
        """Number of frames accumulated."""
        return self._frame_count


def find_minimum_parabola(data):
    """
    Fit a parabola to the given data and find the position of the minimum.
    
    Args:
        data (numpy.ndarray): 1D array of float values representing the data.
        
    Returns:
        float: The position of the minimum of the fitted parabola.
    """
    # Ensure the input data is a 1D numpy array
    data = np.asarray(data).ravel()
    
    # Define the parabola function
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Get the indices of the data points
    x = np.arange(len(data))
    
    # Fit the parabola using non-linear least squares
    try:
        popt, pcov = curve_fit(parabola, x, data, p0=[1, 1, 1])
    except RuntimeError:
        # If the fit fails, try again with a different initial guess
        try:
            popt, pcov = curve_fit(parabola, x, data, p0=[-1, -1, -1])
        except RuntimeError:
            # If the fit still fails, return None
            return None
    
    # Calculate the minimum position
    a, b, c = popt
    if a == 0:
        # If the parabola is a straight line, return None
        return None
    else:
        min_position = -b / (2 * a)
        return min_position

        
def compute_hfd(image):
    """
    Compute the half flux diameter (HFD) of a star image in a 2D array.
    
    Args:
        image (numpy.ndarray): 2D array containing the star image.
        
    Returns:
        float: The half flux diameter (HFD) of the star image.
    """
    # Find the centroid (center of mass) of the star image
    image = image - np.min(image)
    total_flux = np.sum(image)
    y, x = np.indices(image.shape)
    y_centroid = np.sum(y * image) / total_flux
    x_centroid = np.sum(x * image) / total_flux
    
    # Sort the pixel values in descending order
    sorted_pixels = np.sort(image.ravel())[::-1]
    
    # Calculate the cumulative sum of the sorted pixel values
    cumsum = np.cumsum(sorted_pixels)
    
    # Find the radius at which the cumulative sum reaches half of the total flux
    half_flux = total_flux / 2
    idx = np.searchsorted(cumsum, half_flux, side='right')
    radius = np.sqrt((idx - 1) / np.pi)
    
    # The HFD is twice the radius
    hfd = 2 * radius
    
    return hfd
    
def extract_centered_subarray(array, subarray_size):
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2-dimensional")
    
    if subarray_size % 2 == 0:
        raise ValueError("Subarray size must be odd")
    
    # Find the coordinates of the maximum value
    max_coord = np.unravel_index(np.argmax(array), array.shape)

    # Calculate the center of the subarray
    center_x, center_y = max_coord
    
    # Calculate the half-width of the subarray
    half_width = subarray_size // 2

    # Calculate the boundaries of the subarray
    start_x = max(0, center_x - half_width)
    end_x = min(array.shape[0], center_x + half_width + 1)
    start_y = max(0, center_y - half_width)
    end_y = min(array.shape[1], center_y + half_width + 1)

    # Extract the subarray
    subarray = array[start_x:end_x, start_y:end_y]

    # If the subarray is smaller than the desired size due to edge cases, pad it
    if subarray.shape != (subarray_size, subarray_size):
        padded = np.zeros((subarray_size, subarray_size))
        x_offset = half_width - center_x if center_x < half_width else 0
        y_offset = half_width - center_y if center_y < half_width else 0
        padded[x_offset:x_offset+subarray.shape[0], 
               y_offset:y_offset+subarray.shape[1]] = subarray
        return padded
    
    return subarray


    
def fit_gauss_circular(data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D circular gaussian PSF.
    ---------------------
    Inputs

    * data (2D Numpy array) = small subimage
    ---------------------
    Output (list) = list with 6 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm is the gaussian PSF full width half maximum (fit result) in pixels
    ---------------------
    """
    
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height < 200:
        return 0
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    
    #---------------------------------------------------------------------------------
    sig = fwhm / (2.*np.sqrt(2.*np.log(2.)))
    width = 0.5/np.square(sig)
    
    p0 = floor, height, mean_x, mean_y, width

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def gauss(floor, height, mean_x, mean_y, width):        
        return lambda x,y: floor + height*np.exp(-np.abs(width)*((x-mean_x)**2+(y-mean_y)**2))

    def err(p,data):
        return np.ravel(gauss(*p)(*np.indices(data.shape))-data)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = leastsq(err, p0, args=(data), maxfev=100, full_output=True)
    p = result[0]
    # Check if fit converged (ier in 1,2,3,4 means success)
    if result[4] not in (1, 2, 3, 4):
        return 0.0  # Return 0 if fit failed
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2]
    mean_y = p[3]

    sig = np.sqrt(0.5/np.abs(p[4]))
    fwhm = sig * (2.*np.sqrt(2.*np.log(2.)))    
    
    output = [maxi, floor, height, mean_x, mean_y, fwhm]
    return fwhm


def fit_gauss_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical gaussian PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 8 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    ---------------------
    """

    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor
    
    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm
    sig_1 = fwhm_1 / (2.*np.sqrt(2.*np.log(2.)))
    sig_2 = fwhm_2 / (2.*np.sqrt(2.*np.log(2.)))    

    angle = 0.

    p0 = floor, height, mean_x, mean_y, sig_1, sig_2, angle

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def gauss(floor, height, mean_x, mean_y, sig_1, sig_2, angle):
    
        A = (np.cos(angle)/sig_1)**2. + (np.sin(angle)/sig_2)**2.
        B = (np.sin(angle)/sig_1)**2. + (np.cos(angle)/sig_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./(sig_1**2.)-1./(sig_2**2.))

        #do not forget factor 0.5 in exp(-0.5*r**2./sig**2.)    
        return lambda x,y: floor + height*np.exp(-0.5*(A*((x-mean_x)**2)+B*((y-mean_y)**2)+C*(x-mean_x)*(y-mean_y)))

    def err(p,data):
        return np.ravel(gauss(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    
    #angle gives the direction of the p[4]=sig_1 axis, starting from x (vertical) axis, clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):

        fwhm_large = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))    
        angle = np.arctan(np.tan(p[6]))
            
    else:    #then sig_1 is the smallest : we want angle to point to sig_y, the largest
    
        fwhm_large = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))    
        angle = np.arctan(np.tan(p[6]+np.pi/2.))
    
    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]
    return output

def fit_moffat_circular(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D circular moffat PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 7 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm, beta]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm is the gaussian PSF full width half maximum (fit result) in pixels
    - beta is the "beta" parameter of the moffat function
    ---------------------
    """
    
    #---------------------------------------------------------------------------------
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))

    beta = 4
    
    p0 = floor, height, mean_x, mean_y, fwhm, beta

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def moffat(floor, height, mean_x, mean_y, fwhm, beta):
        alpha = 0.5*fwhm/np.sqrt(2.**(1./beta)-1.)    
        return lambda x,y: floor + height/((1.+(((x-mean_x)**2+(y-mean_y)**2)/alpha**2.))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    fwhm = np.abs(p[4])
    beta = p[5]
    
    output = [maxi, floor, height, mean_x, mean_y, fwhm, beta]
    return output

def fit_moffat_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical moffat PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 9 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle, beta]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    - beta is the "beta" parameter of the moffat function    
    ---------------------
    """
    
    #---------------------------------------------------------------------------------
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm

    angle = 0.
    beta = 4
    
    p0 = floor, height, mean_x, mean_y, fwhm_1, fwhm_2, angle, beta

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def moffat(floor, height, mean_x, mean_y, fwhm_1, fwhm_2, angle, beta):
        
        alpha_1 = 0.5*fwhm_1/np.sqrt(2.**(1./beta)-1.)
        alpha_2 = 0.5*fwhm_2/np.sqrt(2.**(1./beta)-1.)
    
        A = (np.cos(angle)/alpha_1)**2. + (np.sin(angle)/alpha_2)**2.
        B = (np.sin(angle)/alpha_1)**2. + (np.cos(angle)/alpha_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./alpha_1**2. - 1./alpha_2**2.)
        
        return lambda x,y: floor + height/((1.+ A*((x-mean_x)**2) + B*((y-mean_y)**2) + C*(x-mean_x)*(y-mean_y))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    beta = p[7]
    
    #angle gives the direction of the p[4]=fwhm_1 axis, starting from x (vertical) axis, clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):

        fwhm_large = np.abs(p[4])
        fwhm_small = np.abs(p[5])
        angle = np.arctan(np.tan(p[6]))
            
    else:    #then fwhm_1 is the smallest : we want angle to point to sig_y, the largest
    
        fwhm_large = np.abs(p[5])
        fwhm_small = np.abs(p[4])
        angle = np.arctan(np.tan(p[6]+np.pi/2.))

    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle, beta]
    return output



from cv2 import medianBlur

import numpy as np
from cv2 import medianBlur

class HighValueFinder:
    def __init__(self, search_box_size=32, blur_size=3):
        self.hint_x = None
        self.hint_y = None
        self.reference_value = None
        self.search_box_size = search_box_size
        self.blur_size = blur_size

    def find_high_value_element(self, array):
        array = array.astype('float32')
        filtered_array = medianBlur(array, self.blur_size)

        if self.hint_x is not None and self.hint_y is not None and self.reference_value is not None:
            # Define the search box boundaries
            x_start = max(0, self.hint_x - self.search_box_size // 2)
            x_end = min(array.shape[1], self.hint_x + self.search_box_size // 2)
            y_start = max(0, self.hint_y - self.search_box_size // 2)
            y_end = min(array.shape[0], self.hint_y + self.search_box_size // 2)
            
            # Extract the search box
            search_area = filtered_array[y_start:y_end, x_start:x_end]
            
            # Find the maximum value within the search box
            local_max = np.max(search_area)
            
            # If the local max is less than half the reference value, do a full scan
            if local_max < 0.5 * self.reference_value:
                return self._full_array_scan(filtered_array)
            
            local_rows, local_cols = np.where(search_area == local_max)
            
            # Translate local coordinates back to global coordinates
            col = local_cols[0] + x_start
            row = local_rows[0] + y_start
        else:
            # If no hint is available, do a full array scan
            col, row, val = self._full_array_scan(filtered_array)
        
        # Update hint and reference value for next call
        self.hint_x, self.hint_y = col, row
        self.reference_value = filtered_array[row, col]
        
        return col, row, filtered_array[row, col]

    def _full_array_scan(self, array):
        rows, cols = np.where(array == np.max(array))
        return cols[0], rows[0], array[rows[0], cols[0]]

    def reset(self):
        self.hint_x = None
        self.hint_y = None
        self.reference_value = None

def find_high_value_element(array, size=3):
    array = array.astype('float32')
    filtered_array = medianBlur(array, size)
    rows, cols = np.where(filtered_array == np.max(filtered_array))
    return cols[0], rows[0], filtered_array[rows[0], cols[0]]


def find_high_value_element(array, size = 3):
  array = array.astype('float32')
  #print(array.dtype)
  filtered_array = medianBlur(array,3)


  #filtered_array = array

  # Find the indices of the maximum value
  row, col = np.where(filtered_array == np.max(filtered_array))


  return col[0], row[0]

  
def compute_centroid(array, x, y):


  # Find the indices and values of the star pixels in the array

  N = 32
  array = array[y - N: y + N, x - N:x + N]

  array = array - (np.min(array) + 1.5 * np.std(array))

  rows, cols = np.where(array > 0.0)
  values = array[rows, cols]
  #print(values.shape)

  # Compute the centroid using a weighted average
  centroid_row = np.sum(rows * values) / np.sum(values)
  centroid_col = np.sum(cols * values) / np.sum(values)
  centroid_value = np.max(values)

  return centroid_col + x - N, centroid_row + y - N, centroid_value


from scipy import ndimage
from skimage.filters import threshold_otsu

def compute_centroid_improved(array, x, y, initial_size=32, final_size=16, iterations=3):
    def iterate_centroid(sub_array, size):
        # Use Otsu's method for adaptive thresholding
        thresh = threshold_otsu(sub_array)
        binary = sub_array > thresh
        
        # Use center of mass for initial estimate
        cy, cx = ndimage.center_of_mass(binary)
        
        # Refine using weighted centroid
        rows, cols = np.where(binary)
        values = sub_array[rows, cols]
        
        # --- FIX: Prevent division by zero on saturated images ---
        sum_of_values = np.sum(values)
        if sum_of_values == 0:
            # If all values are zero (e.g., saturated star after background subtraction),
            # return the geometric center of the sub-array as a fallback.
            return size // 2, size // 2

        centroid_row = np.sum(rows * values) / sum_of_values
        centroid_col = np.sum(cols * values) / sum_of_values
        
        return centroid_col, centroid_row

    current_x, current_y = x, y
    current_size = initial_size

    for _ in range(iterations):
        half_size = current_size // 2
        sub_array = array[int(current_y - half_size):int(current_y + half_size),
                          int(current_x - half_size):int(current_x + half_size)]
        
        dx, dy = iterate_centroid(sub_array, current_size)
        
        current_x += dx - half_size
        current_y += dy - half_size
        
        current_size = max(final_size, current_size // 2)

    # Final refined centroid
    final_half_size = final_size // 2
    final_array = array[int(current_y - final_half_size):int(current_y + final_half_size),
                        int(current_x - final_half_size):int(current_x + final_half_size)]
    
    final_dx, final_dy = iterate_centroid(final_array, final_size)
    
    final_x = current_x + final_dx - final_half_size
    final_y = current_y + final_dy - final_half_size
    
    centroid_value = np.max(final_array)
    
    return final_x, final_y, centroid_value


from scipy.optimize import minimize


def find_optimal_scaling(array1, array2):
    def minimize_std(K):
        # Calculate the difference array
        diff = array1 - array2 * K

        # Calculate the standard deviation of the difference array
        std = np.std(diff)
        #print(std)
        # Return the standard deviation as the optimization target
        return std

    # Initialize the optimization parameters
    x0 = [1.0]  # Initial value for K
    bounds = [(0.0, None)]  # Bounds for K (must be non-negative)

    # Perform the optimization
    result = minimize(minimize_std, x0, bounds=bounds)

    # Extract the optimized value for K
    K_opt = result.x[0]

    return K_opt




class ImageNavigator(QtWidgets.QMainWindow):
    def __init__(self, images, filenames):
        super().__init__()
        self.images = [np.rot90(image) for image in images]  # Rotate each image by 90 degrees
        self.filenames = filenames
        self.currentIndex = 0
        self.histLevels = None  # Store histogram levels
        self.prevTransform = None  # Initialize variable to store previous transformation

        self.initUI()
        
    def click(self, event):
        event.accept()      
        self.pos = event.pos()
        print("click", int(self.pos.x()), int(self.pos.y()))
        x = int(self.pos.x())
        y = int(self.pos.y())
        data = self.images[self.currentIndex][x-10:x+10, y-10:y+10]
        data = data - np.min(data)
        print("max = ", np.max(data))
        print("hfd = ", compute_hfd(data))

        fwhm = fit_gauss_circular(data)
        print("fwhm = ", fwhm)

    def initUI(self):
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.centralWidget.setLayout(self.layout)
        
        self.imageView = pg.ImageView()
        self.layout.addWidget(self.imageView)
        
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.buttonLayout)
        
        self.prevButton = QtWidgets.QPushButton("Previous")
        self.prevButton.clicked.connect(self.prevImage)
        self.buttonLayout.addWidget(self.prevButton)
        
        self.nextButton = QtWidgets.QPushButton("Next")
        self.nextButton.clicked.connect(self.nextImage)
        self.buttonLayout.addWidget(self.nextButton)
        
        self.updateImage()
        self.resize(1600, 1200)  # Set the window size to 1600x1200

        self.toggleFullScreenAction = QtWidgets.QAction("Toggle Full-Screen", self)
        self.toggleFullScreenAction.setShortcut("f")
        self.toggleFullScreenAction.triggered.connect(self.toggleFullScreen)
        self.addAction(self.toggleFullScreenAction)

        self.deleteFileAction = QtWidgets.QAction("Delete File", self)
        self.deleteFileAction.setShortcut("d")
        self.deleteFileAction.triggered.connect(self.deleteFile)
        self.addAction(self.deleteFileAction)

        self.showFilenameAction = QtWidgets.QAction("Show Filename", self)
        self.showFilenameAction.setShortcut("n")
        self.showFilenameAction.triggered.connect(self.showFilename)
        self.addAction(self.showFilenameAction)

        self.imageView.getImageItem().mouseClickEvent = self.click

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def prevImage(self):
        if self.currentIndex > 0:
            self.saveHistogramSettings()
            self.saveViewTransform()
            self.currentIndex -= 1
            self.updateImage()
        
    def nextImage(self):
        if self.currentIndex < len(self.images) - 1:
            self.saveHistogramSettings()
            self.saveViewTransform()
            self.currentIndex += 1
            self.updateImage()
        
    def saveHistogramSettings(self):
        # Save the current histogram settings
        self.histLevels = self.imageView.getHistogramWidget().getLevels()
        
    def saveViewTransform(self):
        # Save the current view transformation matrix
        self.prevTransform = self.imageView.view.getViewBox().transform()

    def updateImage(self):
        self.imageView.setImage(self.images[self.currentIndex], autoHistogramRange=False)
        if self.histLevels:
            self.imageView.setLevels(*self.histLevels)  # Apply saved histogram settings
        self.restoreViewTransform()  # Restore the view state after setting the image

    def restoreViewTransform(self):
        # Restore the saved view transformation if it exists
        if self.prevTransform:
            self.imageView.view.getViewBox().setTransform(self.prevTransform)

    def deleteFile(self):
        if self.currentIndex >= 0 and self.currentIndex < len(self.filenames):
            filename = self.filenames[self.currentIndex]
            #reply = QtWidgets.QMessageBox.question(self, 'Delete File', f'Are you sure you want to delete {filename}?',
            #                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            #if reply == QtWidgets.QMessageBox.Yes:
            try:
                os.remove(filename)
                self.images.pop(self.currentIndex)
                self.filenames.pop(self.currentIndex)
                if self.currentIndex >= len(self.images):
                    self.currentIndex = len(self.images) - 1
                self.updateImage()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to delete file: {e}')

    def showFilename(self):
        if self.currentIndex >= 0 and self.currentIndex < len(self.filenames):
            filename = self.filenames[self.currentIndex]
            print(f"Current file: {filename}")

def loadFITSFiles(fileList):
    images = []
    filenames = []
    for file in fileList:
        data = np.fliplr(fits.getdata(file, ext=0))
        downscaled_data = data[::2, ::2]
        images.append(data)
        filenames.append(file)
    return images, filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FITS Viewer with Navigation")
    parser.add_argument("files", nargs='+', help="List of FITS files to view")
    
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    images, filenames = loadFITSFiles(args.files)
    navigator = ImageNavigator(images, filenames)
    navigator.show()

    sys.exit(app.exec_())




import zmq

class IPC:
    def __init__(self, type = zmq.REQ):
        self.context = zmq.Context()
        self.socket = self.context.socket(type)

        if (type == zmq.REQ):
            self.socket.connect("tcp://localhost:5555")
        else:
            self.socket.bind("tcp://*:5555")


    def get(self):
        count = self.socket.poll(timeout=30)
        if (count != 0):
            obj = self.socket.recv_pyobj()
            return obj
        else:
            return None

    def send(self, msg):
        self.socket.send_pyobj(msg)

    def close(self):
        # Close the socket
        self.socket.close()

        # Terminate the context
        self.context.term()

    def set_val(self, name, val):
        ob = [name, val]
        self.send(ob)
        res = self.get()

    def get_val(self, name):
        ob = [name, -1]
        self.send(ob)
        res = self.get()
        return res


