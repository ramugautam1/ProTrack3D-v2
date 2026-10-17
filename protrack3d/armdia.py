"""The Armadillo / Dia workflow: fixed embryos, one timepoint, several channels in one file.

It replaces the batch scripts the workflow used to be run with (conversion from .czi, segmentation, quantification):

  1. prepare   every .czi or multi-channel TIFF (.tif, .ome.tif, .lsm, in any axis order the file records) under a
               folder: the z slices `z_start` to `z_end` (counted from 1) of the Armadillo channel are
               saved as <name>_ARM.tif and of the Dia channel as <name>_DIA.tif, next to the source file.
  2. segment   <name>_ARM.tif with the chosen model, tiles scaled by the darkest and brightest voxel of the whole
               stack, objects of `size_filter` voxels or fewer removed -> <name>_ARM_SEG.tif.
  3. quantify  Dia and Armadillo inside and outside the mask -> masked images and one summary table for the folder.
               This step runs the original quantification code unchanged.
"""
import os

import numpy as np

from .czi import TIFF_EXTENSIONS, read_multichannel_tif, stem, tif_layout
from .events import Reporter
from .models import load_model, pick_device
from .paths import add_repo_to_path
from .stages import _watching

DERIVED = ('_ARM.tif', '_DIA.tif', '_ARM_SEG.tif', '_ARM_MSK.tif', '_ARM_MP3.tif', '_DIA_MSK.tif', '_DIA_SNR.tif',
           'DIA_MP3.tif', '_cell_outline.tif', '_SEG.tif', 'pct.tif')


def find_sources(root):
    """Source images under root: .czi files, and .tif files that are not outputs of this workflow or outlines.
    A .tif next to a .czi of the same name is a converted copy and is skipped."""
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d.lower() not in ('cell outlines', 'dilated'))
        czi = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.czi')}
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if f.startswith('.') or ext.lower() not in ('.czi',) + TIFF_EXTENSIONS:
                continue
            derived = f.lower().endswith(tuple(d.lower() for d in DERIVED) + tuple(d.lower() + 'f' for d in DERIVED)) or '_boundary_' in f.lower()
            if ext.lower() == '.czi' or (not derived and stem not in czi):
                found.append(os.path.join(folder, f))
    return found


def describe_folder(root):
    """How many source images a folder holds and what the first one looks like, without reading pixel data."""
    sources = find_sources(root)
    info = {'images': len(sources), 'folders': len({os.path.dirname(s) for s in sources}), 'first': None, 'problem': None, 'note': None}
    if sources:
        first = sources[0]
        if first.lower().endswith('.czi'):
            import czifile
            with czifile.CziFile(first) as czi:
                size = dict(zip(czi.axes, czi.shape))
        else:
            try:
                (t, c, z, y, x), note = tif_layout(first)
            except ValueError as e:
                info['problem'] = str(e)
                return info
            size = {'T': t, 'C': c, 'Z': z, 'Y': y, 'X': x}
            info['note'] = note
        info['first'] = {'name': os.path.basename(first), 'channels': int(size.get('C', 1)), 'z': int(size.get('Z', 1)),
                         'y': int(size.get('Y', 0)), 'x': int(size.get('X', 0))}
    return info


def read_stack(path):
    """A source image as (z, channel, y, x). CZI: first scene, one timepoint. TIFF: any axis order the file records
    (ImageJ hyperstack, OME-TIFF, .lsm, RGB), one timepoint."""
    if path.lower().endswith('.czi'):
        import czifile
        with czifile.CziFile(path) as czi:
            data, axes = czi.asarray(), czi.axes
        keep = [i for i, (axis, n) in enumerate(zip(axes, data.shape)) if axis in 'ZCYX' or n > 1]
        data = data.reshape([data.shape[i] for i in keep])
        axes = ''.join(axes[i] for i in keep)
        if set(axes) - set('ZCYX'):
            extra = ', '.join(f'{a}={n}' for a, n in zip(axes, data.shape) if a not in 'ZCYX')
            raise ValueError(f'{os.path.basename(path)} has more than one timepoint or scene ({extra}); this workflow takes single stacks.')
        for axis in 'ZC':
            if axis not in axes:
                data, axes = data[None], axis + axes
        return np.ascontiguousarray(data.transpose([axes.index(a) for a in 'ZCYX']))
    data = read_multichannel_tif(path)[0]  # (t, c, z, y, x)
    if data.shape[0] > 1:
        raise ValueError(f'{os.path.basename(path)} has {data.shape[0]} timepoints; this workflow takes single stacks.')
    return np.ascontiguousarray(data[0].transpose(1, 0, 2, 3))


def remove_small_objects(mask, size_filter):
    """Keep face-connected objects with more than size_filter voxels."""
    import scipy.ndimage as ndi
    labels, _ = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    keep = np.bincount(labels.ravel()) > size_filter
    keep[0] = False
    return keep[labels]


def segment_arm(arm_path, model, device, size_filter=160):
    """<name>_ARM.tif (ZYX) -> <name>_ARM_SEG.tif, the mask as 0 / 65535. Returns (path, number of objects)."""
    import scipy.ndimage as ndi
    import tifffile
    from .segmentation import segment_timepoint
    volume = np.transpose(tifffile.imread(arm_path), (2, 1, 0))  # x, y, z
    mask, _ = segment_timepoint(model, device, volume, normalise='stack', with_features=False)
    mask = remove_small_objects(mask, size_filter)
    out = os.path.splitext(arm_path)[0] + '_SEG.tif'
    tifffile.imwrite(out, (np.transpose(mask, (2, 1, 0)).astype(np.uint16) * 65535), imagej=True, metadata={'axes': 'ZYX'})
    return out, int(ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))[1])


def run(root, model_path, z_start=1, z_end=20, size_filter=160, arm_channel=2, dia_channel=3, arm_only=False, threshold=True,
        csv_name=None, steps=('prepare', 'segment', 'quantify'), device=None, reporter=None):
    """Run the workflow on every source image under `root`. Channels and z slices are counted from 1; slices z_start to
    z_end (both included) are used, or up to the last slice when an image has fewer.

    arm_only: the images hold only Armadillo (its channel is still `arm_channel`); nothing is done for Dia.
    threshold: before measuring, set voxels at or below (background mean + 3 standard deviations) to zero, the
    background being everything outside the Armadillo mask. Applied to Dia, or to Armadillo when arm_only.
    """
    import tifffile
    reporter = reporter or Reporter()
    root = os.path.abspath(root)
    sources = find_sources(root)
    names = [stem(s) for s in sources]
    if not sources:
        raise ValueError(f'No .czi or TIFF (.tif, .tiff, .lsm) images were found under {root}.')
    result = {'folder': root, 'images': len(sources)}

    if 'prepare' in steps:
        z_start, z_end = int(z_start), int(z_end)
        if not 1 <= z_start <= z_end:
            raise ValueError(f'the slice range {z_start} to {z_end} is not valid: slices are counted from 1, and the first cannot come after the last.')
        reporter.stage('preparation', total=len(sources), message=f'{len(sources)} images: slices {z_start} to {z_end} of the Armadillo'
                       + ('' if arm_only else ' and Dia') + ' channels')
        for i, (source, name) in enumerate(zip(sources, names)):
            stack = read_stack(source)
            channels, depth = stack.shape[1], stack.shape[0]
            if z_start > depth:
                raise ValueError(f'{os.path.basename(source)} has {depth} z slices; the range starts at slice {z_start}.')
            for channel, suffix in ((arm_channel, '_ARM.tif'),) + (() if arm_only else ((dia_channel, '_DIA.tif'),)):
                if not 1 <= channel <= channels:
                    raise ValueError(f'{os.path.basename(source)} has {channels} channel(s); channel {channel} was asked for.')
                tifffile.imwrite(name + suffix, stack[z_start - 1:min(z_end, depth), channel - 1], imagej=True, metadata={'axes': 'ZYX'})
            reporter.progress(i + 1, len(sources), os.path.basename(source))
        reporter.stage_done(images=len(sources))

    if 'segment' in steps:
        device = pick_device(device)
        reporter.stage('segmentation', total=len(sources), message=f'{len(sources)} Armadillo stacks on {device.type}, objects of {size_filter} voxels or fewer removed')
        model, meta = load_model(model_path, device)
        objects = []
        for i, name in enumerate(names):
            _, n = segment_arm(name + '_ARM.tif', model, device, size_filter)
            objects.append(n)
            reporter.progress(i + 1, len(sources), f'{os.path.basename(name)}: {n} objects')
        result.update(model=meta.get('name'), device=device.type, objects=objects)
        reporter.stage_done(objects=objects)

    if 'quantify' in steps:
        add_repo_to_path()
        if arm_only:
            import segment.batch_quantify_arm_from_armseg as original
            original.ARM_SUFFIX, original.SEG_SUFFIX = '_ARM.tif', '_ARM_SEG.tif'
            original.ARM_MSK_SUFFIX, original.ARM_T_SUFFIX = '_ARM_MSK.tif', '_ARM_MP3.tif'
            default_name = 'ARM_Summary.csv'
        else:
            import segment.batch_mask_dia_from_armseg as original
            original.DIA_SUFFIX, original.SEG_SUFFIX, original.MSK_SUFFIX = '_DIA.tif', '_ARM_SEG.tif', '_DIA_MSK.tif'
            original.ARM_MSK_SUFFIX, original.ARM_SUFFIX = '_ARM_MSK.tif', '_ARM.tif'
            original.DIA_2_SUFFIX, original.DIA_T_SUFFIX = '_DIA_SNR.tif', '_DIA_MP3.tif'
            default_name = 'ARM_DIA_Summary.csv'
        original.ROOT = root
        original.CSV_OUT = os.path.join(root, csv_name or default_name)
        reporter.stage('quantification', total=len(sources), message='measuring inside and outside the Armadillo mask'
                       + (', with the background threshold' if threshold else ''))
        seen = []

        def on_line(line):
            if line.strip().endswith('_MSK.tif') and line not in seen:
                seen.append(line)
                reporter.progress(len(seen), max(len(sources), len(seen)), os.path.basename(line.strip()))

        with _watching(on_line):
            original.main(threshold=threshold)
        result['table'] = original.CSV_OUT
        reporter.stage_done(table=original.CSV_OUT, rows=len(seen))
    reporter.result(**{k: v for k, v in result.items() if k != 'objects'})
    return result
