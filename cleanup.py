#!/usr/bin/env python3
"""
Remove vertical banding from astronomical images.

Extracts the column-to-column gain variation (banding) by:
1. For each row, divide by a wide smooth to get high-frequency residual
2. Median across rows cancels real structure, keeps banding
3. Correct by dividing out the banding pattern

Real structure spans many columns and is smooth -> removed by the high-pass.
Banding is a fixed per-column gain -> survives the row median.

Usage:
    python cleanup.py input.fits -o cleaned.fits
    python cleanup.py input.fits -o cleaned.fits --kernel 64
"""

import torch
import torch.nn.functional as F
import numpy as np
from astropy.io import fits
import argparse
import time
import sys

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description='Remove vertical banding from astronomical images')
    parser.add_argument('input', help='Input FITS file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output FITS file (default: input_clean.fits)')
    parser.add_argument('--kernel', type=int, default=64,
                        help='Smoothing kernel width in pixels (default: 64). '
                             'Must be larger than banding period, smaller than '
                             'object scale.')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace('.fits', '_clean.fits')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image
    data = fits.getdata(args.input).astype(np.float32)
    h, w = data.shape
    image = torch.tensor(data, device=device)

    print(f"Loaded: {args.input} ({w}x{h})")
    print(f"  Range: [{image.min().item():.6f}, {image.max().item():.6f}]")
    print(f"  Mean:  {image.mean().item():.6f}")

    t0 = time.time()

    # Step 1: For each row, compute smoothed version (1D horizontal Gaussian)
    # This captures real structure but not column-to-column banding
    ksize = args.kernel
    if ksize % 2 == 0:
        ksize += 1
    sigma = ksize / 4.0

    x = torch.arange(ksize, device=device, dtype=torch.float32) - ksize // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 1, -1)  # (out, in, kH, kW)

    # Pad and smooth each row horizontally
    # image shape: (H, W) -> (1, 1, H, W) for conv2d
    img4d = image.unsqueeze(0).unsqueeze(0)
    pad = ksize // 2
    padded = F.pad(img4d, (pad, pad, 0, 0), mode='reflect')
    smoothed = F.conv2d(padded, kernel)  # (1, 1, H, W)
    smoothed = smoothed.squeeze()  # (H, W)

    print(f"  Smoothing kernel: {ksize} pixels (sigma={sigma:.1f})")

    # Step 2: Compute per-row ratio: how each pixel deviates from the smooth
    # Avoid division by zero
    smoothed = smoothed.clamp(min=1e-10)
    ratio = image / smoothed  # (H, W) - should be ~1.0 everywhere

    # Step 3: Median across rows -> the banding pattern
    # Real structure varies row to row, so it cancels in the median.
    # Banding is the same in every row, so it survives.
    banding = torch.median(ratio, dim=0).values  # (W,)

    print(f"  Banding pattern range: [{banding.min().item():.6f}, "
          f"{banding.max().item():.6f}]")
    print(f"  Banding std: {banding.std().item():.6f}")

    # Step 4: Correction = 1 / banding
    # Clamp to reasonable range - columns with no real signal stay at 1.0
    banding = banding.clamp(min=0.5, max=2.0)
    correction = 1.0 / banding
    result = image * correction.unsqueeze(0)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.2f}s")
    print(f"  Correction range: [{correction.min().item():.6f}, "
          f"{correction.max().item():.6f}]")

    # Save
    output_data = result.cpu().numpy().astype(np.float32)
    fits.writeto(args.output, output_data, overwrite=True)
    print(f"Saved: {args.output}")

    # Save correction vector
    corr_file = args.output.replace('.fits', '_corr.fits')
    fits.writeto(corr_file, correction.cpu().numpy().astype(np.float32),
                 overwrite=True)
    print(f"Saved correction vector: {corr_file}")


if __name__ == '__main__':
    main()
