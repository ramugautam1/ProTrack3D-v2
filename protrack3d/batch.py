"""Folder runs: every .tif in a folder, with optional per-image time ranges from batch_times.txt."""
import os

from .segmentation import read_image_info

TIMES_FILE = 'batch_times.txt'


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(('.tif', '.tiff')) and not f.lower().endswith('_seg.tif') and not f.startswith('.'))


def read_times(folder):
    """batch_times.txt lines are `name.tif, start, end` or `all, start, end`. Returns {name or 'all': (start, end)}."""
    times = {}
    path = os.path.join(folder, TIMES_FILE)
    if os.path.isfile(path):
        for line in open(path):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 3:
                try:
                    times[parts[0].lower()] = (int(parts[1]), int(parts[2]))
                except ValueError:
                    pass
    return times


def time_range(image, times):
    return times.get(os.path.basename(image).lower()) or times.get('all') or (1, read_image_info(image)['t'])
