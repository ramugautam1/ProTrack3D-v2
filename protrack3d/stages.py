"""Tracking, family tree and analysis: thin wrappers around the modules at the repository root.

Those modules report progress by printing, so each wrapper watches their output and turns the lines it
recognises into progress events.
"""
import contextlib
import glob
import os
import re
import sys
import time

from .events import Reporter
from .paths import add_repo_to_path
from .segmentation import image_base, segmentation_folder


def tracking_folder(image, seg_folder):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(seg_folder))), image_base(image) + '_TrackingOutput')


class _Watch:
    """Stands in for sys.stdout: passes text through and calls `on_line` for every finished line."""

    def __init__(self, stream, on_line):
        self.stream, self.on_line, self.buffer = stream, on_line, ''

    def write(self, text):
        self.stream.write(text)
        self.buffer += text
        parts = re.split(r'[\r\n]', self.buffer)
        self.buffer = parts.pop()
        for line in parts:
            if line.strip():
                self.on_line(line)
        if self.buffer.strip():
            self.on_line(self.buffer)
        return len(text)

    def flush(self):
        self.stream.flush()


@contextlib.contextmanager
def _watching(on_line):
    old = sys.stdout
    sys.stdout = _Watch(old, on_line)
    try:
        yield
    finally:
        sys.stdout = old


def _headless_plots():
    import matplotlib
    matplotlib.use('Agg')


def remove_working_files(track_folder, seg_folder, start, end):
    """Delete what only tracking needs, once it has finished. Returns the bytes freed.

    That is the per-timepoint folders tracking works in (<tracking folder>/1, /2, ...) and the feature maps
    segmentation wrote for it (<segmentation folder>/Features). Analysis, family trees and the viewer use the
    combined results and never read these. To track the same image again, run segmentation again first.
    """
    import shutil
    freed = 0
    for t in range(start, end + 1):
        folder = os.path.join(track_folder, str(t))
        if os.path.isfile(os.path.join(folder, f'Weights_{t}.nii')) or os.path.isfile(os.path.join(folder, f'Fullsize_{t}.nii')):
            freed += sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
            shutil.rmtree(folder)
    features = os.path.join(seg_folder, 'Features')
    if os.path.isdir(features):
        for name in os.listdir(features):
            if name.startswith('features_') and name.endswith('.npy'):
                freed += os.path.getsize(os.path.join(features, name))
                os.remove(os.path.join(features, name))
        if not os.listdir(features):
            os.rmdir(features)
    return freed


def track(image, seg_folder=None, start=1, frames=None, trackback=2, reporter=None, seed=0, all_voxels=False,
          keep_working_folders=False):
    """Track objects through `frames` segmented timepoints, counted from `start` within the segmented range.

    Tracking compares large objects from a random sample of their voxels. The sample is fixed by `seed`, so a run can
    be repeated exactly; a negative seed gives a different sample every run. `all_voxels` uses every voxel instead.
    The files only tracking needs (its per-timepoint folders and the segmentation feature maps) are deleted when
    tracking has finished, unless keep_working_folders is set.
    """
    add_repo_to_path()
    _headless_plots()
    import pandas as pd
    from myTrackStep1 import myTrackStep1
    from myTrackStep2 import myTrackStep2

    reporter = reporter or Reporter()
    seg_folder = os.path.normpath(seg_folder or segmentation_folder(image))
    if frames is None:
        params = pd.read_csv(os.path.join(seg_folder, 'segmentation_parameters.csv'))
        frames = int(params['endTime'][0]) - int(params['startTime'][0]) + 1
    end = start + frames - 1
    if frames < 2:
        raise ValueError('tracking needs at least 2 segmented timepoints')
    if not os.path.isfile(os.path.join(seg_folder, 'Features', f'features_{start}.npy')) and not os.path.isdir(os.path.join(seg_folder, str(start))):
        raise ValueError('The feature maps of this segmentation are gone: they are deleted after tracking to save space. '
                         'Run segmentation again, then tracking.')
    out = tracking_folder(image, seg_folder)
    os.makedirs(out, exist_ok=True)
    seg_image = glob.glob(os.path.join(seg_folder, '*.nii'))[-1]
    total = 3 * frames  # step 1 and the projections handle every timepoint once; step 2 handles every transition
    reporter.stage('tracking', total=total, message=f'{os.path.basename(image)}, timepoints {start}-{end}')
    state = {'done': 0}

    def on_line(line):
        m = (re.search(r'Processing timepoint : (\d+)/', line) or re.search(r'time point: (\d+) -->', line)
             or re.search(r'^\s*t = (\d+)', line))
        if not m:
            return
        n = int(m.group(1)) - start + 1 if 't = ' not in line else int(m.group(1))
        base = 0 if 'Processing timepoint' in line else frames if 'time point:' in line else 2 * frames
        done = min(total, base + n)
        if done > state['done']:
            state['done'] = done
            phase = 'preparing' if base == 0 else 'linking' if base == frames else 'drawing projections'
            reporter.progress(done, total, phase)

    import numpy as np
    import correlation20220708
    correlation20220708.SAMPLE_LARGE_OBJECTS = not all_voxels
    if seed is not None and seed >= 0:
        np.random.seed(seed)
    with _watching(on_line):
        myTrackStep1(seg_folder + '/', out + '/', start, end, imageNameS=seg_image, imageNameO=image)
        myTrackStep2(seg_op_folder=seg_folder + '/', track_op_folder=out + '/', modelName=os.path.basename(seg_folder),
                     trackbackT=trackback, startpoint=start, endpoint=end - 1, protein1Name='proteinA_',
                     protein2Name='proteinB_', imageNameS=seg_image, imageNameO=image, stime=time.perf_counter())
    result = {'tracking_folder': out, 'frames': frames}
    if not keep_working_folders and os.path.isfile(os.path.join(out, 'TrackedCombined.nii')):
        result['freed_gb'] = round(remove_working_files(out, seg_folder, start, end) / 1e9, 2)
    counts = os.path.join(out, 'eventObjectCount.csv')
    if os.path.isfile(counts):
        df = pd.read_csv(counts)
        result.update(objects_first=int(df['total'].iloc[0]), objects_last=int(df['total'].iloc[-1]),
                      splits=int(df['split'].sum()), merges=int(df['merged'].sum()))
    reporter.stage_done(**result)
    return result


def analyze(image, seg_folder, track_folder, start=1, end=None, reporter=None):
    add_repo_to_path()
    _headless_plots()
    import pandas as pd
    from allAnalysisApp import all_analysis_app

    reporter = reporter or Reporter()
    if end is None:
        params = pd.read_csv(os.path.join(seg_folder, 'segmentation_parameters.csv'))
        end = int(params['endTime'][0]) - int(params['startTime'][0]) + 1
    reporter.stage('analysis', total=None, message=f'{os.path.basename(image)}, timepoints {start}-{end}')
    all_analysis_app(trackedimagepath=os.path.join(track_folder, 'TrackedCombined.nii'), segPath=seg_folder,
                     origImgPath=image, t1=start, t2=end)
    out = os.path.join(track_folder, f'Analysis{start}-{end}')
    result = {'analysis_folder': out, 'files': len(os.listdir(out)) if os.path.isdir(out) else 0}
    reporter.stage_done(**result)
    return result


def _trees_expected(track_folder, ids, complete, partial_merges):
    """How many trees a request will draw, so progress can be shown."""
    from .lineage import Lineage
    lineage = Lineage(track_folder, partial_merges)
    everything = re.sub(r'[^a-zA-Z0-9 ]', '', ids).lower() == 'all'
    wanted = []
    if not everything:
        for part in ids.replace(' ', '').split(','):
            try:
                low, _, high = part.partition('-')
                wanted += range(int(low), int(high or low) + 1)
            except ValueError:
                pass
    if complete:
        if everything:
            return sum(n > 1 for n in lineage.family_size.values())
        return len({lineage.family[i] for i in wanted if i in lineage.family})
    return len(lineage.objects) if everything else len(wanted)


def family_tree(track_folder, out_dir=None, ids='all', reporter=None, complete=False, partial_merges=False,
                overview=False, min_lifetime=1):
    """Family trees for the objects in `ids`.

    The original trees follow one object's future. `complete` draws whole families instead: ancestors, merge partners
    and all their relatives. `overview` adds one picture of every object in the movie.
    """
    add_repo_to_path()
    _headless_plots()

    reporter = reporter or Reporter()
    out_dir = out_dir or os.path.join(track_folder, 'FamilyTrees')
    os.makedirs(out_dir, exist_ok=True)
    table = os.path.join(track_folder, 'Sheet3.csv')
    expected = _trees_expected(track_folder, ids, complete, partial_merges)
    reporter.stage('family tree', total=expected, message=f'IDs: {ids}' + (', complete families' if complete else ''))
    seen = []

    def on_line(line):
        m = re.search(r'Processing ID 0*(\d+)', line)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
            reporter.progress(len(seen), max(expected, len(seen)), f'{"family" if complete else "object"} {m.group(1)}')

    result = {'family_tree_folder': out_dir}
    with _watching(on_line):
        if complete:
            from FamilyTreeComplete import generateCompleteFamilyTree
            result['families'] = len(generateCompleteFamilyTree(table, out_dir, ids, partialMerges=partial_merges))
        else:
            from FamilyTreeWithSplitAndMerge import generateFamilyTree
            generateFamilyTree(excelFile=table, ftFolder=out_dir, tidlist=ids)
        if overview:
            from FamilyTreeComplete import drawAllObjects
            result['overview'] = os.path.join(out_dir, 'AllObjects.png')
            info = drawAllObjects(table, result['overview'], partialMerges=partial_merges, minLifetime=min_lifetime)
            result.update(overview_objects=info['objects'], overview_families=info['families'])
    folder = 'TreeDiagrams_Complete' if complete else 'TreeDiagrams'
    result['diagrams'] = len(glob.glob(os.path.join(glob.escape(out_dir), folder, '*.png')))
    reporter.stage_done(**result)
    return result


def pipeline(image, model_path, out_dir=None, start=1, end=None, trackback=2, steps=('seg', 'track', 'analysis'),
             device=None, reporter=None, seed=0, keep_working_folders=False):
    """Segmentation, tracking and analysis for one image, as the batch page runs them."""
    from .segmentation import segment_image, read_image_info
    reporter = reporter or Reporter()
    results = {}
    seg_folder = segmentation_folder(image, out_dir)
    if 'seg' in steps:
        results['seg'] = segment_image(image, model_path, out_dir, start, end, device=device, reporter=reporter)
    frames = (end or read_image_info(image)['t']) - start + 1
    if 'track' in steps:
        results['track'] = track(image, seg_folder, 1, frames, trackback, reporter, seed, keep_working_folders=keep_working_folders)
    if 'analysis' in steps:
        results['analysis'] = analyze(image, seg_folder, tracking_folder(image, seg_folder), 1, frames, reporter)
    return results
