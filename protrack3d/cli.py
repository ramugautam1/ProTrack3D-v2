"""Command line for the ProTrack3D engine:  python -m protrack3d <command> ...

The app and the Claude skill both drive the pipeline through these commands. Add --json for machine-readable
progress (one JSON object per line, prefixed with '@@PT3D ').
"""
import argparse
import json
import os
import sys
import traceback

from .events import Reporter
from .paths import models_dir


def _model_path(name):
    if os.path.isfile(name):
        return name
    path = os.path.join(models_dir(), name if name.endswith('.pt') else name + '.pt')
    if not os.path.isfile(path):
        raise SystemExit(f'model not found: {name} (looked in {models_dir()})')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='protrack3d', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true', help='print progress as JSON lines')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('info', help='show the shape of an image')
    p.add_argument('image')

    sub.add_parser('models', help='list the installed models')
    sub.add_parser('device', help='show what segmentation will run on')
    sub.add_parser('catalog', help='list the published models and whether each is installed')
    p = sub.add_parser('download-models', help='download published models (default: all that are not installed)')
    p.add_argument('names', nargs='*')

    p = sub.add_parser('import-model', help='turn a converted TensorFlow checkpoint (.npz) into a model file')
    p.add_argument('npz')

    def image_args(p, model=True):
        p.add_argument('image')
        if model:
            p.add_argument('--model', required=True, help='model name or path to a .pt file')
        p.add_argument('--out', help='output folder (default: next to the image)')
        p.add_argument('--start', type=int, default=1, help='first timepoint, counted from 1')
        p.add_argument('--end', type=int, help='last timepoint (default: the last one)')
        p.add_argument('--device', default='auto', help='auto, cuda, mps or cpu')

    p = sub.add_parser('segment', help='segment an image')
    image_args(p)
    p.add_argument('--no-seg-tif', action='store_true', help='do not write the mask as <image>_SEG.tif')

    p = sub.add_parser('train', help='train a segmentation model on a folder of two-channel stacks (image, mask)')
    p.add_argument('folder')
    p.add_argument('--name', help='name of the model (default: the folder name and the date)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--from', dest='base', help='start from this model (fine-tuning) instead of from scratch')
    p.add_argument('--continue', dest='resume', action='store_true', help='carry on training the model called --name')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('training-data', help='show what training on a folder would use')
    p.add_argument('folder')

    p = sub.add_parser('track', help='track the objects of a segmented image')
    p.add_argument('image')
    p.add_argument('--seg-folder', help='the ..._SegmentationOutput/FC-DenseNet folder (default: next to the image)')
    p.add_argument('--start', type=int, default=1, help='first segmented timepoint to track')
    p.add_argument('--frames', type=int, help='number of timepoints to track (default: all segmented)')
    p.add_argument('--trackback', type=int, default=2)
    p.add_argument('--seed', type=int, default=0, help='seed of the random sampling in tracking (default 0, so runs '
                                                       'repeat exactly; -1 for a different sample every run)')
    p.add_argument('--all-voxels', action='store_true', help='compare large objects from every voxel, not a sample')
    p.add_argument('--keep-working-folders', action='store_true', help='do not delete the files only tracking needs (its per-timepoint folders and the feature maps)')

    p = sub.add_parser('analyze', help='plots and tables from a tracked image')
    p.add_argument('image')
    p.add_argument('--seg-folder')
    p.add_argument('--tracking-folder')
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int)

    p = sub.add_parser('tree', help='family trees for tracked objects')
    p.add_argument('tracking_folder')
    p.add_argument('--out')
    p.add_argument('--ids', default='all', help="e.g. 76 | 1,2,13 | 1-29 | all")
    p.add_argument('--complete', action='store_true', help='draw whole families: ancestors, merge partners and all '
                                                           'their relatives, not only the future of each object')
    p.add_argument('--partial-merges', action='store_true', help='with --complete or --overview: also connect objects '
                                                                 'through split-and-merge events')
    p.add_argument('--overview', action='store_true', help='also draw AllObjects.png, one picture of every object')
    p.add_argument('--min-lifetime', type=int, default=1, help='overview: leave out unrelated objects that exist for '
                                                               'fewer timepoints')

    p = sub.add_parser('pipeline', help='segmentation, tracking and analysis for one image')
    image_args(p)
    p.add_argument('--trackback', type=int, default=2)
    p.add_argument('--seed', type=int, default=0, help='seed of the random sampling in tracking (-1: none)')
    p.add_argument('--steps', default='seg,track,analysis')
    p.add_argument('--keep-working-folders', action='store_true', help='do not delete the files only tracking needs (its per-timepoint folders and the feature maps)')

    p = sub.add_parser('armdia', help='Armadillo/Dia workflow on a folder of .czi or .tif stacks (one timepoint each)')
    p.add_argument('folder')
    p.add_argument('--model', required=True)
    p.add_argument('--z-layers', type=int, default=20, help='use the first this-many z slices')
    p.add_argument('--size-filter', type=int, default=160, help='remove objects of this many voxels or fewer')
    p.add_argument('--arm-channel', type=int, default=2, help='channel holding Armadillo, counted from 1')
    p.add_argument('--dia-channel', type=int, default=3, help='channel holding Dia, counted from 1')
    p.add_argument('--arm-only', action='store_true', help='the images hold Armadillo only')
    p.add_argument('--no-threshold', action='store_true', help='measure without the background threshold')
    p.add_argument('--csv', help='name of the summary table (default ARM_DIA_Summary.csv)')
    p.add_argument('--steps', default='prepare,segment,quantify')
    p.add_argument('--device', default='auto')

    p = sub.add_parser('armdia-boundary', help='Armadillo puncta on and off the cell boundary (needs cell-outline images)')
    p.add_argument('folder', help='holds the embryo folders (directly or in "Embryos") and "Cell Outlines"')
    p.add_argument('--outlines', help='the outlines folder, if it is somewhere else')
    p.add_argument('--current-width', type=int, default=6, help='width of the drawn outline, in pixels')
    p.add_argument('--target-width', type=int, default=30, help='width to thicken it to')
    p.add_argument('--overlap', type=float, default=0.30, help='share of a punctum that must lie on the boundary')
    p.add_argument('--z-suffix', default='-20z', help='ending of image names that outline names do not have')

    p = sub.add_parser('batch', help='run the pipeline on every .tif in a folder (time ranges from batch_times.txt)')
    p.add_argument('folder')
    p.add_argument('--model', required=True)
    p.add_argument('--trackback', type=int, default=2)
    p.add_argument('--seed', type=int, default=0, help='seed of the random sampling in tracking (-1: none)')
    p.add_argument('--steps', default='seg,track,analysis')
    p.add_argument('--keep-working-folders', action='store_true', help='do not delete the files only tracking needs (its per-timepoint folders and the feature maps)')
    p.add_argument('--device', default='auto')

    args = parser.parse_args(argv)
    reporter = Reporter(json_mode=args.json)
    try:
        return run(args, reporter)
    except Exception as e:
        if not args.json:
            traceback.print_exc()
        reporter.error(f'{type(e).__name__}: {e}')
        return 1


def run(args, reporter):
    from . import stages
    from .segmentation import read_image_info, segment_image, segmentation_folder

    if args.command == 'info':
        print(json.dumps(read_image_info(args.image), indent=None if args.json else 2))
    elif args.command == 'models':
        from .models import list_models
        print(json.dumps(list_models(models_dir()), indent=None if args.json else 2))
    elif args.command == 'catalog':
        from .catalog import load_catalog
        print(json.dumps([{k: m[k] for k in ('name', 'size', 'installed')} for m in load_catalog()['models']],
                         indent=None if args.json else 2))
    elif args.command == 'download-models':
        from .catalog import download
        download(args.names, reporter)
    elif args.command == 'device':
        from .models import pick_device
        device = pick_device().type
        print(json.dumps({'device': device, 'label': {'cuda': 'NVIDIA GPU', 'mps': 'Apple GPU', 'cpu': 'CPU'}[device]}))
    elif args.command == 'import-model':
        from .convert_tf import import_npz
        path, meta = import_npz(args.npz)
        reporter.result(model=path, **meta)
    elif args.command == 'segment':
        segment_image(args.image, _model_path(args.model), args.out, args.start, args.end, device=args.device,
                      save_seg_tif=not args.no_seg_tif, reporter=reporter)
    elif args.command == 'train':
        from .training import train
        train(args.folder, args.name, args.epochs, _model_path(args.base) if args.base else None, args.resume,
              args.batch_size, args.device, args.seed, reporter)
    elif args.command == 'training-data':
        from .training import describe_folder
        print(json.dumps(describe_folder(args.folder), indent=None if args.json else 2))
    elif args.command == 'track':
        stages.track(args.image, args.seg_folder, args.start, args.frames, args.trackback, reporter, args.seed,
                     args.all_voxels, args.keep_working_folders)
    elif args.command == 'analyze':
        seg = args.seg_folder or segmentation_folder(args.image)
        stages.analyze(args.image, seg, args.tracking_folder or stages.tracking_folder(args.image, seg),
                       args.start, args.end, reporter)
    elif args.command == 'tree':
        stages.family_tree(args.tracking_folder, args.out, args.ids, reporter, args.complete, args.partial_merges,
                           args.overview, args.min_lifetime)
    elif args.command == 'pipeline':
        stages.pipeline(args.image, _model_path(args.model), args.out, args.start, args.end, args.trackback,
                        tuple(args.steps.split(',')), args.device, reporter, args.seed, args.keep_working_folders)
    elif args.command == 'armdia':
        from .armdia import run as run_armdia
        run_armdia(args.folder, _model_path(args.model), args.z_layers, args.size_filter, args.arm_channel, args.dia_channel,
                   args.arm_only, not args.no_threshold, args.csv, tuple(args.steps.split(',')), args.device, reporter)
    elif args.command == 'armdia-boundary':
        from .armdia_boundary import run as run_boundary
        run_boundary(args.folder, None, args.outlines, args.current_width, args.target_width, args.overlap, args.z_suffix, reporter)
    elif args.command == 'batch':
        from .batch import list_images, read_times, time_range
        times, failed = read_times(args.folder), []
        for image in list_images(args.folder):
            start, end = time_range(image, times)
            try:
                stages.pipeline(image, _model_path(args.model), None, start, end, args.trackback,
                                tuple(args.steps.split(',')), args.device, reporter, args.seed, args.keep_working_folders)
            except Exception as e:
                failed.append(os.path.basename(image))
                reporter.error(f'{os.path.basename(image)}: {type(e).__name__}: {e}')
        reporter.result(images=len(list_images(args.folder)), failed=failed)
        return 1 if failed else 0
    return 0
