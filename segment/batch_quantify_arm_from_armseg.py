# batch_quantify_arm_from_armseg.py
#
# ARM-only version:
#   - walks ROOT for *_ARM.tif
#   - reads matching *_ARM_SEG.tif (nonzero = foreground)
#   - optional MP3 threshold (bg_mean + 3*bg_std) applied to ARM
#   - writes *_ARM_MSK.tif (masked ARM) and, if thresholding, *_ARM_MP3.tif
#   - writes one CSV row of ARM metrics per image
import os
import csv
import numpy as np
import tifffile
import scipy.ndimage as ndi


def threshold_by_mp3(data, mask):
    """
    Zero out voxels <= (bg_mean + 3*bg_std), where background is
    everything outside the segmentation mask.
    Returns a NEW array; does not modify the input.
    """
    print(data.shape, mask.shape)
    bool_mask = mask.astype(bool)
    bg_voxels = data[~bool_mask]
    bg_mean = np.mean(bg_voxels)
    bg_std = np.std(bg_voxels)
    val = bg_mean + 3 * bg_std

    out = data.copy()
    out[out <= val] = 0
    return out


def count_objects(mask: np.ndarray) -> int:
    """
    mask: boolean or binary array, ZYX or TZYX.
    Objects are face-connected, as in the segmentation size filter.
    """
    structure = ndi.generate_binary_structure(3, 1)  # objects are joined through faces only
    if mask.ndim == 4:
        total = 0
        for t in range(mask.shape[0]):
            _, n = ndi.label(mask[t], structure=structure)
            total += n
        return total
    elif mask.ndim == 3:
        _, n = ndi.label(mask, structure=structure)
        return n
    else:
        raise ValueError(f"Unsupported mask ndim: {mask.ndim}")


####################################

def read_tif(path):
    a = tifffile.imread(path)
    a = np.asarray(a)
    if a.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D. Got {a.shape}")
    return a


def fg_bg_intensity_density(arm_image, mask_path, fg_is_nonzero=True):
    """
    density = total_intensity / num_pixels
    Foreground determined by mask (default: mask > 0).
    Background is the complement.
    Returns a dict with sums, counts, densities, and fg/bg ratio.
    """
    img = arm_image.astype(np.float64, copy=False)
    m = read_tif(mask_path)

    if img.shape != m.shape:
        raise ValueError(f"Shape mismatch: image={img.shape}, mask={m.shape}")

    fg = (m > 0) if fg_is_nonzero else (m == 0)
    bg = ~fg

    fg_n = int(fg.sum())
    bg_n = int(bg.sum())
    if fg_n == 0:
        raise ValueError("Foreground has 0 pixels (mask > 0 is empty).")
    if bg_n == 0:
        raise ValueError("Background has 0 pixels (mask covers entire image).")

    fg_sum = float(img[fg].sum())
    bg_sum = float(img[bg].sum())

    fg_density = fg_sum / fg_n
    bg_density = bg_sum / bg_n

    return {
        "fg_pixels": fg_n,
        "bg_pixels": bg_n,
        "fg_total_intensity": fg_sum,
        "bg_total_intensity": bg_sum,
        "fg_density": fg_density,
        "bg_density": bg_density,
        "fg_bg_density_ratio": (fg_density / bg_density) if bg_density != 0 else np.inf,
        "fg_minus_bg_density": fg_density - bg_density,
    }


####################################

def main(threshold=False):
    rows = []
    for dirpath, _, filenames in os.walk(ROOT):
        for fn in filenames:
            if not fn.endswith(ARM_SUFFIX):
                continue
            # skip derived files that also happen to end in _ARM.tif variants
            if fn.endswith((ARM_MSK_SUFFIX, ARM_T_SUFFIX)):
                continue

            arm_path = os.path.join(dirpath, fn)
            base = fn[:-len(ARM_SUFFIX)]  # remove "_ARM.tif"
            seg_path = os.path.join(dirpath, base + SEG_SUFFIX)
            arm_msk_path = os.path.join(dirpath, base + ARM_MSK_SUFFIX)
            arm_t_path = os.path.join(dirpath, base + ARM_T_SUFFIX)

            if not os.path.exists(seg_path):
                print(f"[SKIP] No segmentation found for {arm_path}")
                continue

            # --- stage 1: read ARM + SEG, write ARM_MSK ---
            arm = tifffile.imread(arm_path)
            seg = tifffile.imread(seg_path)

            if threshold:
                arm_t = threshold_by_mp3(arm, seg)
            else:
                arm_t = arm.copy()

            mask = seg > 0

            num_objects = count_objects(mask)

            arm_msk = np.where(mask, arm_t, 0).astype(arm.dtype, copy=False)

            print(arm_msk_path)

            # Keep axes consistent with your current usage (ZYX)
            tifffile.imwrite(arm_msk_path, arm_msk, imagej=True,
                             metadata={"axes": "ZYX"})
            if threshold:
                tifffile.imwrite(arm_t_path, arm_t, imagej=True,
                                 metadata={"axes": "ZYX"})

            # --- stage 2: compute metrics for CSV ---

            arm_min = int(arm_t.min())
            arm_max = int(arm_t.max())

            fg_pixels = int(np.count_nonzero(mask))

            # IMPORTANT: force uint64 accumulator to avoid overflow
            sum_arm_msk = int(np.sum(arm_msk, dtype=np.uint64))
            avg_arm_msk = (sum_arm_msk / fg_pixels) if fg_pixels > 0 else 0.0

            sum_arm_msk_po = sum_arm_msk / num_objects if num_objects > 0 else 0

            if arm_max > arm_min and fg_pixels > 0:
                # normalize ARM using global min/max, then sum masked values
                arm_norm01 = (arm_t.astype(np.float32) - arm_min) / (arm_max - arm_min)
                sum_arm_norm01_msk = float(np.sum(arm_norm01[mask], dtype=np.float64))
                avg_arm_norm01_msk = sum_arm_norm01_msk / fg_pixels
            else:
                sum_arm_norm01_msk = 0.0
                avg_arm_norm01_msk = 0.0

            # ---- fg/bg intensity density on ARM ----
            res = fg_bg_intensity_density(
                arm_image=arm_t,
                mask_path=seg_path,
            )

            rows.append({
                "fg_pixels": res["fg_pixels"],
                "bg_pixels": res["bg_pixels"],
                "num_objects": num_objects,

                # --- ARM foreground / background intensity metrics ---
                "fg_tot_int_arm": res["fg_total_intensity"],
                "bg_tot_int_arm": res["bg_total_intensity"],
                "fg_per_pix_arm": res["fg_density"],
                "bg_per_pix_arm": res["bg_density"],
                "fg_bg_per_pix_ratio_arm": res["fg_bg_density_ratio"],
                "fg_bg_per_pix_diff_arm": res["fg_minus_bg_density"],

                # --- masked ARM metrics ---
                "sum_arm_msk": sum_arm_msk,
                "avg_arm_msk": avg_arm_msk,
                "sum_arm_msk_po": sum_arm_msk_po,
                "arm_min": arm_min,
                "arm_max": arm_max,
                "sum_arm_norm01_msk": sum_arm_norm01_msk,
                "avg_arm_norm01_msk": avg_arm_norm01_msk,

                "path_arm": arm_path,
                "path_arm_seg": seg_path,
                "path_arm_msk": arm_msk_path,
            })

    cols = [
        "fg_pixels",
        "bg_pixels",
        "num_objects",

        "fg_tot_int_arm",
        "bg_tot_int_arm",
        "fg_per_pix_arm",
        "bg_per_pix_arm",
        "fg_bg_per_pix_ratio_arm",
        "fg_bg_per_pix_diff_arm",

        "sum_arm_msk",
        "avg_arm_msk",
        "sum_arm_msk_po",
        "arm_min",
        "arm_max",
        "sum_arm_norm01_msk",
        "avg_arm_norm01_msk",

        "path_arm",
        "path_arm_seg",
        "path_arm_msk",
    ]

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[DONE] {len(rows)} rows written to {CSV_OUT}")


if __name__ == "__main__":
    # python batch_quantify_arm_from_armseg.py <folder>   walks the folder for *_ARM.tif with *_ARM_SEG.tif
    import sys

    threshold = True
    ROOT = sys.argv[1]
    CSV_OUT = os.path.join(ROOT, "ARM_Summary.csv")

    ARM_SUFFIX = "_ARM.tif"          # input ARM
    SEG_SUFFIX = "_ARM_SEG.tif"      # input mask (nonzero = foreground)
    ARM_MSK_SUFFIX = "_ARM_MSK.tif"  # output masked ARM
    ARM_T_SUFFIX = "_ARM_MP3.tif"    # output thresholded ARM (only if threshold=True)

    main(threshold=threshold)
