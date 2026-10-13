"""Segmentation with the PyTorch FC-DenseNet.

Same tiling, per-tile normalisation and post-processing as the original (TensorFlow) ProTrack3D, done in memory. Images of
any x and y size are accepted; nothing is resized.
Outputs go to <out>/<image>_SegmentationOutput/FC-DenseNet/ in the layout tracking and analysis expect, except that
the 64 feature channels are stored once per timepoint (Features/features_<t>.npy) instead of four files per tile.
"""
import os

import numpy as np

from .events import Reporter
from .models import load_model, pick_device
from .paths import add_repo_to_path

TILE = 32
MODEL_FOLDER = 'FC-DenseNet'
FEATURE_CHANNELS = 64  # tracking uses the first 64 channels of the network's final feature stack


def image_base(image):
    return os.path.splitext(os.path.basename(image))[0]


def segmentation_folder(image, out_dir=None):
    return os.path.join(out_dir or os.path.dirname(os.path.abspath(image)), image_base(image) + '_SegmentationOutput', MODEL_FOLDER)


def read_image_info(image):
    """Shape of an input image without loading it. Sizes are reported as x, y, z, t."""
    if image.lower().endswith('.nii'):
        import nibabel as nib
        shape = nib.load(image).shape
        x, y, z = shape[:3]
        t, axes = (shape[3] if len(shape) > 3 else 1), 'XYZT'[:len(shape)]
        dtype = str(nib.load(image).get_data_dtype())
    else:
        import tifffile
        with tifffile.TiffFile(image) as tif:
            series = tif.series[0]
            shape, axes, dtype = series.shape, series.axes, str(series.dtype)
        if 'C' in axes or len(shape) not in (3, 4):
            raise ValueError(f'{os.path.basename(image)} has axes {axes} {shape}. ProTrack3D needs a single-channel '
                             f'image: ZYX for one timepoint or TZYX for a time-lapse.')
        t = shape[0] if len(shape) == 4 else 1
        z, y, x = shape[-3:]
    return {'path': os.path.abspath(image), 'x': int(x), 'y': int(y), 'z': int(z), 't': int(t), 'axes': axes,
            'dtype': dtype, 'size_mb': round(os.path.getsize(image) / 1e6, 1),
            'edges_extended': bool(x % TILE or y % TILE), 'tiles_per_timepoint': -(-x // TILE) * -(-y // TILE)}


def load_image(image):
    """The image as (x, y, z, t)."""
    if image.lower().endswith('.nii'):
        import nibabel as nib
        vol = np.asarray(nib.load(image).dataobj).astype(np.int16)
        return vol[..., None] if vol.ndim == 3 else vol
    import tifffile
    read_image_info(image)  # refuses multi-channel images with a clear message
    im = tifffile.imread(image)
    if im.ndim == 3:
        im = im[None]
    return np.transpose(im, (3, 2, 1, 0))


def _tiles(volume):
    x, y, z = volume.shape
    return volume.reshape(x // TILE, TILE, y // TILE, TILE, z).transpose(0, 2, 1, 3, 4).reshape(-1, TILE, TILE, z)


def _untile(tiles, x, y):
    rest = tiles.shape[3:]
    order = (0, 2, 1, 3) + tuple(range(4, 4 + len(rest)))
    return tiles.reshape(x // TILE, y // TILE, TILE, TILE, *rest).transpose(order).reshape(x, y, *rest)


def segment_timepoint(model, device, volume, batch_size=16, normalise='tile', with_features=True):
    """volume: (x, y, z) intensities for one timepoint. Returns (cell mask (x, y, z) bool, features (x, y, z, 64)).

    normalise: 'tile' scales every 32 x 32 tile by its own darkest and brightest voxel (the standard pipeline);
    'stack' scales all tiles by the darkest and brightest voxel of the whole stack (the Armadillo/Dia workflow).
    with_features=False skips the feature maps, which only tracking needs, and returns None for them.

    The network works on 32 x 32 tiles. When x or y is not a multiple of 32 the far edges are extended with a mirror
    image of the picture, and the result is cut back to the original size. Tiles that hold only real pixels give
    exactly what they would give without the extension.
    """
    import torch
    x, y, z = volume.shape
    v = volume.astype(np.float64)
    if v.mean() < 0:
        v = v + 32768
    if x % TILE or y % TILE:
        v = np.pad(v, ((0, -x % TILE), (0, -y % TILE), (0, 0)), mode='symmetric')
    px, py = v.shape[:2]
    whole = (float(volume.min()), float(volume.max())) if normalise == 'stack' else None
    tiles = _tiles(v)
    if whole is None:
        lo = tiles.min(axis=(1, 2, 3), keepdims=True)
        hi = tiles.max(axis=(1, 2, 3), keepdims=True)
        flat = (hi == lo).ravel()  # a tile with no contrast holds no objects; treat it as background
        norm = np.float32((tiles - lo) / np.where(hi == lo, 1, hi - lo) * 0.5 + 0.5)
    else:
        lo, hi = whole
        flat = np.zeros(len(tiles), bool)
        norm = np.float32((tiles - lo) / (hi - lo) * 0.5 + 0.5) if hi > lo else np.full(tiles.shape, 0.5, np.float32)
    batch = torch.from_numpy(norm)[:, None]
    cell, feats = [], []
    with torch.inference_mode():
        for i in range(0, len(batch), batch_size):
            logits, stack = model(batch[i:i + batch_size].to(device))
            cell.append((logits.argmax(1) == 0).cpu())
            if with_features:
                feats.append(stack[:, :FEATURE_CHANNELS].permute(0, 2, 3, 4, 1).float().cpu())
    cell = torch.cat(cell).numpy()
    cell[flat] = False
    if not with_features:
        return _untile(cell, px, py)[:x, :y], None
    feats = torch.cat(feats).numpy()
    feats[flat] = 0
    return _untile(cell, px, py)[:x, :y], _untile(feats, px, py)[:x, :y]


def _size_histogram(labeled, path):
    import matplotlib.pyplot as plt
    counts = np.unique(labeled, return_counts=True)[1][1:]
    unique, n = np.unique(counts, return_counts=True)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter([0] * len(counts), counts, c=[n[np.where(unique == v)[0][0]] for v in counts],
                          cmap=plt.get_cmap('rainbow'), marker='s', s=10, edgecolor='none', alpha=0.7)
    plt.colorbar(scatter, label='Count')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.title('Size Histogram')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(path)
    plt.close()


def segment_image(image, model_path, out_dir=None, start=1, end=None, device=None, batch_size=16, save_seg_tif=True,
            reporter=None):
    """Segment timepoints start..end (1-based, inclusive) of `image`. Returns a dict of what was written."""
    import nibabel as nib
    import pandas as pd
    add_repo_to_path()
    from segment.pruneAndFilter import pruneAndFilter
    from segment.saveTif import saveTif

    reporter = reporter or Reporter()
    info = read_image_info(image)
    end = info['t'] if end is None else end
    if not 1 <= start <= end <= info['t']:
        raise ValueError(f'time range {start}-{end} is outside the image, which has {info["t"]} timepoint(s)')
    n_t = end - start + 1
    folder = segmentation_folder(image, out_dir)
    os.makedirs(os.path.join(folder, 'CombinedSO'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Features'), exist_ok=True)

    device = pick_device(device)
    reporter.stage('segmentation', total=n_t, message=f'{os.path.basename(image)}, timepoints {start}-{end}, on {device.type}')
    model, meta = load_model(model_path, device)
    volume = load_image(image)[..., start - 1:end]
    x, y, z = volume.shape[:3]

    pd.DataFrame({'startTime': [int(start)], 'endTime': [int(end)], 'Model': [os.path.abspath(model_path)]}).to_csv(
        os.path.join(folder, 'segmentation_parameters.csv'))

    # Normalised copy of the input, kept because tracking reads the image size from it.
    lo, hi = float(volume.min()), float(volume.max())
    scaled = np.empty(volume.shape, np.float32)
    for t in range(n_t):
        scaled[..., t] = (volume[..., t].astype(np.float64) - lo) / ((hi - lo) or 1) * 32768 - 49152
    nib.save(nib.Nifti1Image(scaled, np.eye(4)), os.path.join(folder, image_base(image) + '.nii'))
    del scaled

    labeled = np.zeros((x, y, z, n_t), np.uint16)
    for t in range(n_t):
        cell, feats = segment_timepoint(model, device, volume[..., t], batch_size)
        np.save(os.path.join(folder, 'Features', f'features_{t + 1}.npy'), feats)
        labeled[..., t] = pruneAndFilter(cell.astype(np.uint16), remove_deg=(0, 2), protect_lt=40, remove_small_le=8, axes='XYZ')
        reporter.progress(t + 1, n_t, f'timepoint {start + t}: {int(labeled[..., t].max())} objects')

    nib.save(nib.Nifti1Image(labeled, np.eye(4)), os.path.join(folder, 'CombinedSO', 'CombinedSOLabeled.nii'))
    objects_per_timepoint = [int(labeled[..., t].max()) for t in range(n_t)]
    labeled[labeled > 0] = 65535
    nib.save(nib.Nifti1Image(labeled, np.eye(4)), os.path.join(folder, 'CombinedSO', 'CombinedSO.nii'))
    seg_tif = None
    if save_seg_tif:
        saveTif(labeled, image)
        seg_tif = os.path.join(os.path.dirname(image), image_base(image) + '_SEG.tif')
    if n_t == 1:
        import skimage.measure
        single = np.uint16(skimage.measure.label(labeled.astype(bool), connectivity=1))
        nib.save(nib.Nifti1Image(single, np.eye(4)), os.path.join(folder, 'CombinedSO', 'LabeledSO.nii'))
        _size_histogram(single, os.path.join(folder, 'CombinedSO', 'SizeHistogram.png'))

    result = {'segmentation_folder': folder, 'image': image, 'seg_tif': seg_tif, 'timepoints': n_t,
              'start': start, 'end': end, 'model': meta.get('name'), 'device': device.type,
              'objects_per_timepoint': objects_per_timepoint}
    reporter.stage_done(**result)
    return result
