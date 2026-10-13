"""Reading Zeiss .czi files for display: every channel, any number of timepoints."""
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


def read_multichannel_tif(path):
    """A .tif with a channel axis (for example a training stack: image and mask) as (t, channel, z, y, x)."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data, axes = series.asarray(), series.axes
    if len(axes) != data.ndim or set(axes) - set('TCZYX'):
        raise ValueError(f'{os.path.basename(path)} has axes {axes}, which the viewer cannot show.')
    for axis in 'TCZ':
        if axis not in axes:
            data, axes = data[None], axis + axes
    return np.ascontiguousarray(data.transpose([axes.index(a) for a in 'TCZYX'])), [''] * data.shape[axes.index('C')], []
