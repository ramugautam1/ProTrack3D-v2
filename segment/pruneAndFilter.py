import numpy as np
import scipy.ndimage as ndi

def pruneAndFilter(
    bin_vol: np.ndarray,
    protect_lt: int = 40,
    remove_deg=(0, 1, 2),
    remove_small_le: int = 8,
    connectivity: int = 1,   # 1 => 6-connectivity in 3D
    dtype_out=np.uint16,
    axes: str = "XYZ",       # input axis order: "XYZ" or "ZYX"
) -> np.ndarray:
    """
    Input: 3D binary volume in axis order given by `axes` ("XYZ" or "ZYX")
    Output: labeled pruned mask in the SAME axis order as input.
    """

    if bin_vol.ndim != 3:
        raise ValueError(f"Expected 3D volume. Got shape {bin_vol.shape}")
    if axes not in ("XYZ", "ZYX"):
        raise ValueError("axes must be 'XYZ' or 'ZYX'")

    # Convert to ZYX internally (so neighbor logic matches your code)
    if axes == "XYZ":
        bin_zyx = np.transpose(bin_vol, (2, 1, 0))  # XYZ -> ZYX
    else:
        bin_zyx = bin_vol

    v = (bin_zyx > 0)
    structure = ndi.generate_binary_structure(3, connectivity)

    # 1) label
    lbl, num = ndi.label(v, structure=structure)
    if num == 0:
        out_zyx = np.zeros_like(bin_zyx, dtype=dtype_out)
        return np.transpose(out_zyx, (2, 1, 0)) if axes == "XYZ" else out_zyx

    sizes = np.bincount(lbl.ravel())

    # 2) protect < protect_lt
    small_ids = np.flatnonzero((sizes < protect_lt) & (np.arange(sizes.size) != 0))
    protect = np.isin(lbl, small_ids) if small_ids.size else np.zeros_like(v, dtype=bool)

    # 3) face-degree (6-neighbor)
    vv = v.astype(np.uint8)
    n = np.zeros_like(vv, dtype=np.uint8)

    n[:-1,:,:] += vv[1:,:,:]
    n[1:,:,:]  += vv[:-1,:,:]
    n[:,:-1,:] += vv[:,1:,:]
    n[:,1:,:]  += vv[:,:-1,:]
    n[:,:,:-1] += vv[:,:,1:]
    n[:,:,1:]  += vv[:,:,:-1]

    rm = np.zeros_like(v, dtype=bool)
    for d in remove_deg:
        rm |= v & (n == d)
    rm &= ~protect

    v_pruned = v.copy()
    v_pruned[rm] = False

    # 4) relabel after prune
    lbl2, num2 = ndi.label(v_pruned, structure=structure)
    if num2 == 0:
        out_zyx = np.zeros_like(bin_zyx, dtype=dtype_out)
        return np.transpose(out_zyx, (2, 1, 0)) if axes == "XYZ" else out_zyx

    sizes2 = np.bincount(lbl2.ravel())

    # 5) remove tiny objects <= remove_small_le
    tiny_ids = np.flatnonzero((sizes2 <= remove_small_le) & (np.arange(sizes2.size) != 0))
    if tiny_ids.size:
        v_pruned[np.isin(lbl2, tiny_ids)] = False

    # 6) final relabel
    lbl_final_zyx, _ = ndi.label(v_pruned, structure=structure)
    out_zyx = lbl_final_zyx.astype(dtype_out)

    # Convert back to requested axis order
    return np.transpose(out_zyx, (2, 1, 0)) if axes == "XYZ" else out_zyx