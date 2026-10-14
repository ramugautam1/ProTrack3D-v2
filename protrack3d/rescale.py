"""Saving an image with a brightness and contrast setting applied to its pixel values.

A .tif or .nii is written anew. A .czi is copied and only the pixel values in the copy are changed, so all of its
metadata stay; it can also be saved as a .tif.

Every value is rescaled so that `low` becomes 0 and `high` becomes the largest value the image type holds (255 for
8-bit, 65535 for 16-bit, 1.0 for floating point), which is what ImageJ's Apply does. Values outside the range are
clipped, so this cannot be undone: the saved image no longer holds the measured intensities.
"""
import gc
import os

import numpy as np

from .events import Reporter


def _top(dtype):
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def _scale(values, low, high, dtype):
    scaled = np.clip((values.astype(np.float32) - low) / max(high - low, 1e-9), 0, 1) * _top(dtype)
    return np.rint(scaled).astype(dtype) if np.issubdtype(dtype, np.integer) else scaled.astype(dtype)


CZI_PIXELS = {'Gray8': np.uint8, 'Gray16': np.dtype('<u2'), 'Gray32Float': np.dtype('<f4')}


def czi_problem(path):
    """Why this .czi cannot be saved as a .czi with new pixel values, or None when it can: every tile has to be stored
    uncompressed with one grey value per pixel, so the new values can be written over the old ones in a copy."""
    import czifile
    with czifile.CziFile(path) as czi:
        for entry in czi.subblock_directory:
            if entry.compression != 0:
                return 'its image data are compressed'
            if entry.pixel_type not in CZI_PIXELS:
                return f'its pixels are {entry.pixel_type}, not grey values'
    return None


def _rescale_czi(image, out, ranges, reporter):
    """Copy the .czi and rescale the pixel values of every tile in the copy. Everything else in the file (all metadata,
    every scene, the pyramid levels) is kept as it was."""
    import shutil

    import czifile
    problem = czi_problem(image)
    if problem:
        raise ValueError(f'{os.path.basename(image)} cannot be saved as a .czi because {problem}. Save it as a .tif instead.')
    with czifile.CziFile(image) as czi:
        tiles = []
        for entry in czi.subblock_directory:
            channel = next((d.start for d in entry.dimension_entries if d.dimension == 'C'), 0)
            segment = entry.data_segment()
            tiles.append((channel, segment.data_offset, segment.data_size, CZI_PIXELS[entry.pixel_type]))
    first = min(t[0] for t in tiles)
    channels = max(t[0] for t in tiles) - first + 1
    ranges = [tuple(r) for r in ranges] * (channels if len(ranges) == 1 else 1)
    if len(ranges) != channels:
        raise ValueError(f'the image has {channels} channels but {len(ranges)} ranges were given')
    part = out + '.part'
    shutil.copyfile(image, part)
    try:
        with open(part, 'r+b') as f:
            for i, (channel, offset, size, dtype) in enumerate(tiles):
                f.seek(offset)
                values = np.frombuffer(f.read(size), dtype)
                f.seek(offset)
                f.write(_scale(values, *ranges[channel - first], dtype).astype(dtype).tobytes())
                reporter.progress(i + 1, len(tiles), f'tile {i + 1} of {len(tiles)}')
    except BaseException:
        os.remove(part)
        raise
    os.replace(part, out)
    return channels


def _tif_details(path):
    """What a rewritten .tif should keep from the original: pixel size and time step."""
    import tifffile
    kept = {'metadata': {}}
    with tifffile.TiffFile(path) as tif:
        axes = tif.series[0].axes
        for key in ('spacing', 'unit', 'finterval'):
            if key in (tif.imagej_metadata or {}):
                kept['metadata'][key] = tif.imagej_metadata[key]
        try:
            page = tif.pages[0]
            if 'XResolution' in page.tags and 'YResolution' in page.tags:
                kept['resolution'] = tuple(page.tags[k].value[0] / page.tags[k].value[1] for k in ('XResolution', 'YResolution'))
                kept['resolutionunit'] = page.tags['ResolutionUnit'].value if 'ResolutionUnit' in page.tags else None
        except (ZeroDivisionError, TypeError, IndexError):
            kept.pop('resolution', None)
    return axes, kept


def _rescale_nii(image, out, ranges, reporter):
    import nibabel as nib
    source = nib.load(image)
    data = np.asarray(source.dataobj)
    (low, high), result = ranges[0], np.empty_like(data)
    frames = data.shape[3] if data.ndim == 4 else 1
    for t in range(frames):
        if data.ndim == 4:
            result[..., t] = _scale(data[..., t], low, high, data.dtype)
        else:
            result[...] = _scale(data, low, high, data.dtype)
        reporter.progress(t + 1, frames, f'timepoint {t + 1}')
    part = out[:-4] + '.part.nii'
    nib.save(nib.Nifti1Image(result, source.affine, source.header), part)
    del source, data
    os.replace(part, out)


def rescale(image, out, ranges, reporter=None):
    """Write `image` to `out` with each channel rescaled from its (low, high) in `ranges` to the full range of the
    image type. `out` may be `image` itself; the original is then replaced once the new file is complete."""
    import tifffile
    reporter = reporter or Reporter()
    image, out = os.path.abspath(image), os.path.abspath(out)
    kind = image.lower()
    wanted = '.nii' if kind.endswith('.nii') else ('.tif', '.tiff', '.czi') if kind.endswith('.czi') else ('.tif', '.tiff')
    if not out.lower().endswith(wanted):
        raise ValueError(f'the saved image has to be a {wanted if isinstance(wanted, str) else " or ".join(wanted)} file')
    reporter.stage('saving image', message=os.path.basename(out))
    if kind.endswith('.czi') and out.lower().endswith('.czi'):
        channels = _rescale_czi(image, out, ranges, reporter)
        result = {'image': out, 'replaced': out == image, 'channels': channels}
        reporter.stage_done(**result)
        return result
    if kind.endswith('.nii'):
        _rescale_nii(image, out, ranges, reporter)
        result = {'image': out, 'replaced': out == image}
        reporter.stage_done(**result)
        return result

    kept, raw = {'metadata': {}}, None
    if kind.endswith('.czi'):
        from .czi import read_czi
        data = read_czi(image)[0]
    else:
        axes, kept = _tif_details(image)
        if 'C' in axes:
            from .czi import read_multichannel_tif
            data = read_multichannel_tif(image)[0]
        else:
            try:
                raw = tifffile.memmap(image, mode='r')
            except Exception:
                raw = tifffile.imread(image)
            if raw.ndim not in (3, 4):
                raise ValueError(f'{os.path.basename(image)} has axes {axes}; expected ZYX or TZYX')
            data = raw.reshape((raw.shape[0] if raw.ndim == 4 else 1, 1) + raw.shape[-3:])
    t, c, z, y, x = data.shape
    ranges = [tuple(r) for r in ranges] * (c if len(ranges) == 1 else 1)
    if len(ranges) != c:
        raise ValueError(f'the image has {c} channels but {len(ranges)} ranges were given')
    dtype = data.dtype

    def planes():
        for ti in range(t):
            for zi in range(z):
                for ci in range(c):
                    yield _scale(np.asarray(data[ti, ci, zi]), *ranges[ci], dtype)
                if t == 1:
                    reporter.progress(zi + 1, z, f'slice {zi + 1}')
            if t > 1:
                reporter.progress(ti + 1, t, f'timepoint {ti + 1}')

    shape, axes_out = [t, z, c, y, x], 'TZCYX'
    for axis, size in (('C', c), ('T', t)):  # a single channel or timepoint is not written as an axis
        if size == 1:
            del shape[axes_out.index(axis)]
            axes_out = axes_out.replace(axis, '')
    imagej = dtype in (np.uint8, np.uint16, np.float32)  # the types ImageJ's own format can hold
    metadata = dict(kept['metadata'], axes=axes_out)
    if imagej and c == 1:
        metadata.update(min=0.0, max=_top(dtype))
    options = {k: kept[k] for k in ('resolution', 'resolutionunit') if kept.get(k) is not None}
    part = out + '.part'
    try:
        tifffile.imwrite(part, planes(), shape=tuple(shape), dtype=dtype, imagej=imagej, metadata=metadata,
                         photometric='minisblack', **options)
    except BaseException:
        if os.path.isfile(part):
            os.remove(part)
        raise
    del data, raw  # the original must not be held open when it is the file being replaced
    gc.collect()
    os.replace(part, out)
    result = {'image': out, 'replaced': out == image, 'channels': c, 'timepoints': t}
    reporter.stage_done(**result)
    return result
