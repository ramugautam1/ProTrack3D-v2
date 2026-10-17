"""Reading multi-channel images: Zeiss .czi files, and TIFFs (ImageJ hyperstacks, OME-TIFF, .lsm, RGB stacks) in
whatever axis order they record."""
import os
import xml.etree.ElementTree as ET

import numpy as np


def read_czi(path):
    """Returns (data as (t, channel, z, y, x), channel names, notes).

    A file with several scenes or tiles is shown by its first one; `notes` says so.
    """
    import czifile
    with czifile.CziFile(path) as czi:
        data, axes = czi.asarray(), czi.axes
        try:
            xml = czi.metadata(raw=True)
        except Exception:
            xml = None
    notes = []
    for i in reversed(range(len(axes))):
        if axes[i] not in 'TCZYX':
            if data.shape[i] > 1:
                notes.append(f'only the first of {data.shape[i]} along "{axes[i]}" is shown')
            data = np.take(data, 0, axis=i)
    axes = ''.join(a for a in axes if a in 'TCZYX')
    for axis in 'TCZ':
        if axis not in axes:
            data, axes = data[None], axis + axes
    data = np.ascontiguousarray(data.transpose([axes.index(a) for a in 'TCZYX']))
    names = []
    if xml:
        try:
            for channel in ET.fromstring(xml).findall('.//Information/Image/Dimensions/Channels/Channel'):
                fluor = channel.findtext('Fluor')
                names.append(fluor or channel.get('Name') or '')
        except ET.ParseError:
            pass
    names = (names + [''] * data.shape[1])[:data.shape[1]]
    return data, names, notes


TIFF_EXTENSIONS = ('.tif', '.tiff', '.lsm')  # .ome.tif ends in .tif; .lsm is Zeiss's older TIFF-based format
FEW_CHANNELS = 6  # an unlabelled axis this short, next to a longer one, is taken to be the channels


def _plan(shape, axes, name):
    """How to turn a TIFF series into (t, channel, z, y, x): the axes after renaming, which axes to drop, and a note
    when the order had to be guessed. Works on the shape alone, so it can describe a file without reading it."""
    axes, shape, note = list(axes.upper()), list(shape), None
    if len(axes) != len(shape):
        raise ValueError(f'{name}: its axes ({"".join(axes)}) do not match its shape {tuple(shape)}.')
    if 'S' in axes:  # samples per pixel (an RGB image): the channels, unless the file also has a channel axis
        s = axes.index('S')
        if 'C' in axes and shape[axes.index('C')] > 1 and shape[s] > 1:
            raise ValueError(f'{name} has both channels and colour samples; save it from Fiji as a plain hyperstack.')
        if 'C' in axes:
            axes[axes.index('C')] = '-'
        axes[s] = 'C'
    unknown = [i for i, a in enumerate(axes) if a not in 'TCZYX-' and shape[i] > 1]
    if len(unknown) == 1 and 'C' in axes and 'Z' not in axes:
        # the channels are known (an RGB or planar stack); the one remaining axis of pages can only be z
        axes[unknown[0]], unknown = 'Z', []
    if unknown and len(axes) == 4 and axes[2:] == ['Y', 'X'] and len(unknown) <= 2 and 'C' not in axes and 'Z' not in axes:
        # a 4-D file saved without saying which axis is which: the short one is the channels, the long one z
        a, b = shape[0], shape[1]
        if b <= FEW_CHANNELS < a or (b <= FEW_CHANNELS and b < a):
            axes[0], axes[1], note = 'Z', 'C', f'the file does not say which axis holds the channels; read as ZCYX ({b} channels)'
        elif a <= FEW_CHANNELS < b or (a <= FEW_CHANNELS and a < b):
            axes[0], axes[1], note = 'C', 'Z', f'the file does not say which axis holds the channels; read as CZYX ({a} channels)'
        unknown = [i for i, a in enumerate(axes) if a not in 'TCZYX-' and shape[i] > 1]
    if unknown:
        what = ', '.join(f'{axes[i]}={shape[i]}' for i in unknown)
        raise ValueError(f'{name} has an axis ProTrack3D cannot place ({what}). Open it in Fiji, make it a hyperstack '
                         f'(Image > Hyperstacks > Stack to Hyperstack) with the right number of channels, and save it as TIFF.')
    drop = [i for i, a in enumerate(axes) if a not in 'TCZYX' and shape[i] == 1 or a == '-']
    return axes, drop, note


def _arranged(data, axes, drop):
    keep = [i for i in range(len(axes)) if i not in drop]
    data = data.reshape([data.shape[i] for i in keep])
    axes = ''.join(axes[i] for i in keep)
    for axis in 'TCZ':
        if axis not in axes:
            data, axes = data[None], axis + axes
    return np.ascontiguousarray(data.transpose([axes.index(a) for a in 'TCZYX']))


def stem(path):
    """The path without its extension; for an .ome.tif without the .ome as well, so outputs read <name>_ARM.tif."""
    base = os.path.splitext(path)[0]
    return base[:-4] if base.lower().endswith('.ome') else base


def tif_layout(path):
    """(t, channels, z, y, x) of a TIFF without reading its pixels, and the note from _plan."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape, axes = series.shape, series.axes
    order, drop, note = _plan(shape, axes, os.path.basename(path))
    sizes = {a: n for a, n, i in zip(order, shape, range(len(order))) if i not in drop}
    return tuple(int(sizes.get(a, 1)) for a in 'TCZYX'), note


def _channel_names(tif, channels):
    """Channel names from OME metadata, when the file has them."""
    names = []
    try:
        if tif.is_ome and tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            names = [c.get('Name') or '' for c in root.iter() if c.tag.endswith('}Channel')]
    except ET.ParseError:
        pass
    return (names + [''] * channels)[:channels]


def read_multichannel_tif(path):
    """A TIFF with several channels (or one), in whatever axis order the file records -- ImageJ hyperstacks, OME-TIFF,
    .lsm, RGB stacks -- as ((t, channel, z, y, x), channel names, notes)."""
    import tifffile
    name = os.path.basename(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        order, drop, note = _plan(series.shape, series.axes, name)
        data = _arranged(series.asarray(), order, drop)
        names = _channel_names(tif, data.shape[1])
    return data, names, [note] if note else []
