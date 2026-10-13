"""Convert a TensorFlow ProTrack3D checkpoint to a PyTorch model file, in two steps:

  1. in an environment with TensorFlow:   python -m protrack3d.convert_tf dump <checkpoint folder> <out.npz>
  2. in the ProTrack3D environment:       python -m protrack3d import-model <out.npz>
"""
import glob
import json
import os
import re
import sys

import numpy as np


def read_training_log(folder):
    """Pull the summary lines the TensorFlow training wrote to logs.txt."""
    info = {}
    path = os.path.join(folder, 'logs.txt')
    if os.path.isfile(path):
        text = open(path).read()
        m = re.search(r'Highest IoU:\s*([0-9.]+) at epoch (\d+)', text)
        if m:
            info['best_val_iou'], info['best_epoch'] = float(m.group(1)), int(m.group(2))
        m = re.search(r'Training Started:\s*([0-9-]+)', text)
        if m:
            info['trained_on'] = m.group(1)
    return info


def dump_checkpoint(folder, npz_path):
    import tensorflow as tf
    index = glob.glob(os.path.join(folder, '*.ckpt.index'))
    if len(index) != 1:
        raise SystemExit(f'expected one .ckpt.index file in {folder}, found {len(index)}')
    reader = tf.train.load_checkpoint(index[0][:-len('.index')])
    names = [n for n in reader.get_variable_to_shape_map()
             if n.split('/')[-1] in ('kernel', 'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance') and 'Adam' not in n]
    meta = dict(read_training_log(folder), source_checkpoint=os.path.basename(os.path.normpath(folder)),
                converted_from='tensorflow')
    np.savez(npz_path, __meta__=json.dumps(meta), **{n: reader.get_tensor(n) for n in names})
    print(f'wrote {npz_path}: {len(names)} tensors from {meta["source_checkpoint"]}')


def import_npz(npz_path, out_path=None):
    from .models import save_model
    from .network import FCDenseNet103, load_tf_variables
    from .paths import models_dir
    data = np.load(npz_path)
    meta = json.loads(str(data['__meta__']))
    variables = {k: data[k] for k in data.files if k != '__meta__'}
    model = load_tf_variables(FCDenseNet103(), variables)
    out_path = out_path or os.path.join(models_dir(), meta['source_checkpoint'] + '.pt')
    meta.update(classes=['Cell', 'Background'], patch_xy=32, feature_channels=64)
    return out_path, save_model(out_path, model, meta)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'dump':
        dump_checkpoint(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
