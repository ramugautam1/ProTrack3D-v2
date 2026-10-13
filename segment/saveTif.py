import numpy as np
import os
import tifffile


def saveTif(seg_tif, image):  # image is actually the path of original image
    # If singleton T, drop it -> (X, Y, Z)
    if seg_tif.ndim == 4 and seg_tif.shape[-1] == 1:
        seg_tif = seg_tif[..., 0]

    orig_dir = os.path.dirname(image)
    orig_base = os.path.splitext(os.path.basename(image))[0]
    seg_tif_path = os.path.join(orig_dir, orig_base + "_SEG.tif")

    if seg_tif.ndim == 3:
        # (X, Y, Z) -> (Z, Y, X)
        seg_tif_out = np.transpose(seg_tif, (2, 1, 0))
        tifffile.imwrite(
            seg_tif_path,
            seg_tif_out.astype(np.uint16),
            imagej=True,
            metadata={"axes": "ZYX"},
        )
    elif seg_tif.ndim == 4:
        # (X, Y, Z, T) -> (T, Z, Y, X)
        seg_tif_out = np.transpose(seg_tif, (3, 2, 1, 0))
        tifffile.imwrite(
            seg_tif_path,
            seg_tif_out.astype(np.uint16),
            imagej=True,
            metadata={"axes": "TZYX"},
        )
    else:
        raise ValueError(f"Unexpected segmentation array shape for TIF export: {seg_tif.shape}")