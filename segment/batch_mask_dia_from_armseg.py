# batch_mask_dia_from_armseg_min.py
import os
import csv
import numpy as np
import tifffile
import copy
import scipy.ndimage as ndi




def threshold_by_mp3(data, mask):
    print(data.shape, mask.shape)
    bool_mask = mask.astype(bool)
    bg_voxels = data[~bool_mask]
    bg_mean = np.mean(bg_voxels)
    bg_std = np.std(bg_voxels)
    val = bg_mean + 3* bg_std

    data[data<= val] = 0
    return data

def count_objects(mask: np.ndarray) -> int:
    """
    mask: boolean or binary array
          ZYX or TZYX
    Objects are face-connected, as in the segmentation size filter.
    """
    if mask.ndim == 4:
        total = 0
        for t in range(mask.shape[0]):
            structure = ndi.generate_binary_structure(3, 1)  # objects are joined through faces only
            _, n = ndi.label(mask[t], structure=structure)
            total += n
        return total
    elif mask.ndim == 3:
        structure = ndi.generate_binary_structure(3, 1)  # objects are joined through faces only
        _, n = ndi.label(mask, structure=structure)
        return n
    else:
        raise ValueError(f"Unsupported mask ndim: {mask.ndim}")

def normalize_by_snr(data, mask, scale_factor=1000):
    """
    Converts 3D SNR map to uint16.
    SNR of 0 (background mean) becomes 0.
    SNR of 1.0 becomes 1000.
    """
    bool_mask = mask.astype(bool)

    # 1. Calculate stats
    bg_voxels = data[~bool_mask]
    bg_mean = np.mean(bg_voxels)
    bg_std = np.std(bg_voxels)

    # 2. Calculate SNR in float
    snr_map = (data.astype(np.float32) - bg_mean ) / bg_std

    # 3. Handle Negatives & Scaling
    # Clip at 0 so we don't wrap around in uint16
    snr_map = np.maximum(snr_map, 0)

    # Scale up (e.g., SNR 15.5 becomes 15500)
    uint_map = snr_map * scale_factor
    # 4. Final Clip & Cast
    # Ensure we don't exceed uint16 max (65535)
    return np.clip(uint_map, 0, 65535).astype(np.uint16)


def normalize_by_snr_plus(data, mask, scale_factor=1000):
    """
    Converts 3D SNR map to uint16.
    SNR of 0 (background mean) becomes 0.
    SNR of 1.0 becomes 1000.
    """
    bool_mask = mask.astype(bool)

    # 1. Calculate stats
    bg_voxels = data[~bool_mask]
    bg_mean = np.mean(bg_voxels)
    bg_std = np.std(bg_voxels)

    # 2. Calculate SNR in float
    snr_map = (data.astype(np.float32) - (bg_mean + 3*bg_std) ) / bg_std

    print(f'{bg_mean:.2f}  {bg_std:.2f}')

    # 3. Handle Negatives & Scaling
    # Clip at 0 so we don't wrap around in uint16
    snr_map = np.maximum(snr_map, 0)

    # Scale up (e.g., SNR 15.5 becomes 15500)
    uint_map = snr_map * scale_factor

    # 4. Final Clip & Cast
    # Ensure we don't exceed uint16 max (65535)
    return np.clip(uint_map, 0, 65535).astype(np.uint16)


####################################

def read_tif(path):
    a = tifffile.imread(path)
    a = np.asarray(a)
    if a.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D. Got {a.shape}")
    return a

def fg_bg_intensity_density(dia_image, mask_path, fg_is_nonzero=True):
    """
    density = total_intensity / num_pixels
    Foreground determined by mask (default: mask > 0).
    Background is the complement.
    Returns a dict with sums, counts, densities, and fg/bg ratio.
    """
    img = dia_image.astype(np.float64, copy=False)
    m   = read_tif(mask_path)

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
            if not fn.endswith(DIA_SUFFIX):
                continue

            dia_path = os.path.join(dirpath, fn)
            base = fn[:-len(DIA_SUFFIX)]  # remove "_DIA.tif"
            seg_path = os.path.join(dirpath, base + SEG_SUFFIX)
            msk_path = os.path.join(dirpath, base + MSK_SUFFIX)

            arm_path = os.path.join(dirpath, base + ARM_SUFFIX)
            arm_msk_path = os.path.join(dirpath, base + ARM_MSK_SUFFIX)

            dia_2_path = os.path.join(dirpath, base + DIA_2_SUFFIX)
            dia_t_path = os.path.join(dirpath, base + DIA_T_SUFFIX)

            # --- stage 1: read DIA + SEG, write DIA_MSK ---
            dia = tifffile.imread(dia_path)          # guaranteed same size as seg
            seg = tifffile.imread(seg_path)
            arm = tifffile.imread(arm_path)

            threshold=threshold


            if threshold==True:
                dia_t = copy.deepcopy(dia)
                dia_t = threshold_by_mp3(dia, seg)
            else:
                dia_t =  copy.deepcopy(dia)

            mask = seg > 0

            num_objects = count_objects(mask)

            dia_msk = np.where(mask, dia_t, 0).astype(dia.dtype, copy=False)
            arm_msk = np.where(mask, arm, 0).astype(dia.dtype, copy=False)

            print(msk_path)


            # Keep axes consistent with your current usage (ZYX)
            tifffile.imwrite(msk_path, dia_msk, imagej=True, metadata={"axes": "ZYX"})
            tifffile.imwrite(arm_msk_path, arm_msk, imagej=True, metadata={"axes":"ZYX"})
            tifffile.imwrite(dia_2_path, dia, imagej=True, metadata={"axes": "ZYX"})
            tifffile.imwrite(dia_t_path, dia_t, imagej=True, metadata={"axes": "ZYX"})



            # --- stage 2: read DIA_MSK, compute metrics for CSV ---

            dia_min = int(dia_t.min())
            dia_max = int(dia_t.max())

            arm_min = int(arm.min())
            arm_max = int(arm.max())

            fg_pixels = int(np.count_nonzero(mask))

            # IMPORTANT: force uint64 accumulator to avoid overflow
            sum_dia_msk = int(np.sum(dia_msk, dtype=np.uint64))
            avg_dia_msk = (sum_dia_msk / fg_pixels) if fg_pixels > 0 else 0.0

            sum_arm_msk = int(np.sum(arm_msk, dtype=np.uint64))
            avg_arm_msk = (sum_arm_msk / fg_pixels) if fg_pixels > 0 else 0.0

            sum_dia_msk_po = sum_dia_msk/num_objects if num_objects>0 else 0
            sum_arm_msk_po = sum_arm_msk/num_objects if num_objects>0 else 0

            if dia_max > dia_min and fg_pixels > 0:
                # normalize DIA using global min/max, then sum masked normalized values
                dia_norm01 = (dia_t.astype(np.float32) - dia_min) / (dia_max - dia_min)
                sum_dia_norm01_msk = float(np.sum(dia_norm01[mask], dtype=np.float64))
                avg_dia_norm01_msk = (sum_dia_norm01_msk / fg_pixels)
            else:
                sum_dia_norm01_msk = 0.0
                avg_dia_norm01_msk = 0.0
            ####################################
            if arm_max > arm_min and fg_pixels > 0:
                # normalize DIA using global min/max, then sum masked normalized values
                arm_norm01 = (arm.astype(np.float32) - arm_min) / (arm_max - arm_min)
                sum_arm_norm01_msk = float(np.sum(arm_norm01[mask], dtype=np.float64))
                avg_arm_norm01_msk = (sum_arm_norm01_msk / fg_pixels)
            else:
                sum_arm_norm01_msk = 0.0
                avg_arm_norm01_msk = 0.0

            if sum_arm_msk > 0:
                ratio = sum_dia_msk/sum_arm_msk
            else:
                ratio=0

            # ---- usage ----
            res = fg_bg_intensity_density(
                dia_image=dia_t,
                mask_path=seg_path,
            )

            # rename / extract fields from res
            fg_pixels = res["fg_pixels"]
            bg_pixels = res["bg_pixels"]

            fg_tot_int_dia = res["fg_total_intensity"]
            bg_tot_int_dia = res["bg_total_intensity"]

            fg_per_pix_dia = res["fg_density"]
            bg_per_pix_dia = res["bg_density"]

            fg_bg_per_pix_ratio_dia = res["fg_bg_density_ratio"]
            fg_bg_per_pix_diff_dia = res["fg_minus_bg_density"]

            rows.append({
                "fg_pixels": fg_pixels,
                "bg_pixels": bg_pixels,
                "num_objects": num_objects,

                # --- DIA foreground / background intensity metrics ---
                "fg_tot_int_dia": fg_tot_int_dia,
                "bg_tot_int_dia": bg_tot_int_dia,
                "fg_per_pix_dia": fg_per_pix_dia,
                "bg_per_pix_dia": bg_per_pix_dia,
                "fg_bg_per_pix_ratio_dia": fg_bg_per_pix_ratio_dia,
                "fg_bg_per_pix_diff_dia": fg_bg_per_pix_diff_dia,

                # --- existing fields (unchanged) ---
                "sum_dia_msk": sum_dia_msk,
                "avg_dia_msk": avg_dia_msk,
                "sum_dia_msk_po": sum_dia_msk_po,
                "dia_min": dia_min,
                "dia_max": dia_max,
                "sum_dia_norm01_msk": sum_dia_norm01_msk,
                "avg_dia_norm01_msk": avg_dia_norm01_msk,

                "sum_arm_msk": sum_arm_msk,
                "avg_arm_msk": avg_arm_msk,
                "sum_arm_msk_po": sum_arm_msk_po,
                "arm_min": arm_min,
                "arm_max": arm_max,
                "sum_arm_norm01_msk": sum_arm_norm01_msk,
                "avg_arm_norm01_msk": avg_arm_norm01_msk,

                "ratio": ratio,

                "path_dia": dia_path,
                "path_arm_seg": seg_path,
                "path_dia_msk": msk_path,
            })

    cols = [
        "fg_pixels",
        "bg_pixels",
        "num_objects",

        "fg_tot_int_dia",
        "bg_tot_int_dia",
        "fg_per_pix_dia",
        "bg_per_pix_dia",
        "fg_bg_per_pix_ratio_dia",
        "fg_bg_per_pix_diff_dia",

        "sum_dia_msk",
        "avg_dia_msk",
        "sum_dia_msk_po",
        "dia_min",
        "dia_max",
        "sum_dia_norm01_msk",
        "avg_dia_norm01_msk",

        "sum_arm_msk",
        "avg_arm_msk",
        "sum_arm_msk_po",
        "arm_min",
        "arm_max",
        "sum_arm_norm01_msk",
        "avg_arm_norm01_msk",

        "ratio",

        "path_dia",
        "path_arm_seg",
        "path_dia_msk",
    ]

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    # python batch_mask_dia_from_armseg.py <folder>   walks the folder for *_DIA.tif with *_ARM.tif and *_ARM_SEG.tif
    import sys

    threshold = True
    ROOT = sys.argv[1]
    CSV_OUT = os.path.join(ROOT, "ARM_DIA_Summary.csv")

    DIA_SUFFIX = "_DIA.tif"  # input DIA
    SEG_SUFFIX = "_ARM_SEG.tif"  # input mask (nonzero = foreground)
    MSK_SUFFIX = "_DIA_MSK.tif"  # output masked DIA
    ARM_MSK_SUFFIX = "_ARM_MSK.tif"
    ARM_SUFFIX = "_ARM.tif"
    DIA_2_SUFFIX = "_DIA_SNR.tif"
    DIA_T_SUFFIX = "_DIA_MP3.tif"

    main(threshold=threshold)
