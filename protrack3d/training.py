"""Training the segmentation model in PyTorch.

A port of the training code of the original (TensorFlow) ProTrack3D: the same tiles (32 x 32, every 24 pixels, each
also flipped in z), the same per-tile normalisation, loss, learning-rate schedule and validation score. Differences:

  * nothing is written into the training folder (the TensorFlow version filled an `aug` folder with tile files);
  * the class weights come from the training masks themselves, not from a fixed set of reference pictures, and are stored in the
    model file;
  * the tiles are shuffled with a fixed seed, so a training run can be repeated.

As in the TensorFlow training, batch normalisation runs on its stored statistics and dropout is off.
"""
import csv
import datetime
import math
import os
import random
import time

import numpy as np

from .events import Reporter
from .models import load_model, pick_device, save_model
from .paths import models_dir

TILE, STEP = 32, 24
VALIDATION_SHARE = 0.15
VALIDATION_TILES = 200
LOSS_SCALE = 100


def learning_rate(epoch, fresh):
    """1e-3 for 25 epochs, then 1e-4, 1e-5 from epoch 50 and 1e-6 from epoch 100; a tenth of that when the training
    starts from an existing model."""
    rate = 1e-3 if epoch < 25 else 1e-4 if epoch < 50 else 1e-5 if epoch < 100 else 1e-6
    return rate if fresh else 0.1 * rate


def list_stacks(folder):
    """Training stacks in a folder, in name order. The last 15 % (at least one) are kept for validation."""
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))
    n_val = int(math.ceil(len(files) * VALIDATION_SHARE))
    return files[:len(files) - n_val], files[len(files) - n_val:]


def read_stack(path):
    """A training stack as (image, mask), each (z, y, x). The file holds two channels: the image, then the mask."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data, axes = series.asarray(), series.axes
    if data.ndim != 4 or 2 not in data.shape[:2]:
        raise ValueError(f'{os.path.basename(path)} has shape {data.shape}. A training stack needs z slices and two '
                         f'channels (ZCYX): channel 1 the image, channel 2 the mask.')
    if axes[:2] == 'CZ' or (data.shape[0] == 2 and data.shape[1] != 2):
        data = data.swapaxes(0, 1)
    return data[:, 0], data[:, 1] > 0


def describe_folder(folder):
    """What training on this folder would use, without loading the pixel data."""
    import tifffile
    train, val = list_stacks(folder)
    info = {'training_files': [os.path.basename(f) for f in train], 'validation_files': [os.path.basename(f) for f in val],
            'tiles_train': 0, 'tiles_val': 0, 'z': None, 'problem': None}
    if not train:
        info['problem'] = 'Training needs at least two stacks: one is always kept for validation.'
        return info
    for group, key in ((train, 'tiles_train'), (val, 'tiles_val')):
        for path in group:
            with tifffile.TiffFile(path) as tif:
                shape = tif.series[0].shape
            if len(shape) != 4:
                info['problem'] = f'{os.path.basename(path)} is not a two-channel z stack (its shape is {shape}).'
                return info
            z, y, x = (shape[0] if shape[1] == 2 else shape[1]), shape[2], shape[3]
            if info['z'] not in (None, z):
                info['problem'] = f'The stacks do not all have the same number of z slices ({info["z"]} and {z}).'
                return info
            info['z'] = z
            info[key] += 2 * max(0, (y - TILE) // STEP + 1) * max(0, (x - TILE) // STEP + 1)
    return info


def make_tiles(image, mask):
    """Cut one stack into training tiles. Returns (images (n, x, y, z) float32 normalised per tile, masks (n, x, y, z) bool)."""
    images, masks = [], []
    z, h, w = image.shape
    for y in range(0, h - TILE + 1, STEP):
        for x in range(0, w - TILE + 1, STEP):
            for flip in (slice(None), slice(None, None, -1)):  # each tile, and the same tile upside down in z
                tile = image[flip, y:y + TILE, x:x + TILE].transpose(2, 1, 0).astype(np.float64)
                lo, hi = tile.min(), tile.max()
                images.append(np.float32((tile - lo) / (hi - lo) * 0.5 + 0.5 if hi > lo else np.full(tile.shape, 0.5)))
                masks.append(mask[flip, y:y + TILE, x:x + TILE].transpose(2, 1, 0))
    return images, masks


def load_tiles(files):
    images, masks = [], []
    for path in files:
        tile_images, tile_masks = make_tiles(*read_stack(path))
        images += tile_images
        masks += tile_masks
    if images and len({t.shape for t in images}) > 1:
        raise ValueError('The training stacks do not all have the same number of z slices.')
    return np.stack(images), np.stack(masks)


def class_weights(masks):
    """Weights for (cell, background), inversely proportional to how many voxels each has; they add up to 1."""
    cell = float(masks.sum())
    background = float(masks.size - cell)
    if cell == 0:
        raise ValueError('The training masks are empty: no voxel is marked as an object.')
    weights = np.array([masks.size / cell, masks.size / background])
    return weights / weights.sum()


def loss_function(logits, target, weights):
    """Cross-entropy per voxel, weighted by the class of the voxel, as in the TensorFlow training. target: 0 cell, 1 background."""
    import torch.nn.functional as F
    return (LOSS_SCALE * F.cross_entropy(logits, target, reduction='none') * weights[target]).mean()


def glorot_init(model):
    """Start the network the way Keras does: Glorot-uniform kernels and zero biases."""
    import torch.nn as nn
    for conv in model.convs:
        nn.init.xavier_uniform_(conv.weight)
        nn.init.zeros_(conv.bias)


def validate(model, device, images, masks, batch_size=16):
    """Mean accuracy and mean IoU of the object class over the validation tiles (IoU as (overlap + 1) / (union + 1))."""
    import torch
    accuracy, iou = [], []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size])[:, None].to(device)
            cell = (model(x)[0].argmax(1) == 0).cpu().numpy()
            truth = masks[i:i + batch_size]
            for p, t in zip(cell, truth):
                accuracy.append(float((p == t).mean()))
                iou.append((np.logical_and(p, t).sum() + 1) / (np.logical_or(p, t).sum() + 1))
    return float(np.mean(accuracy)), float(np.mean(iou))


def train(folder, name=None, epochs=100, base_model=None, continue_training=False, batch_size=8, device=None,
          seed=0, reporter=None):
    """Train on the stacks in `folder` and save the model as <models folder>/<name>.pt.

    base_model: path of a model to start from (fine-tuning). continue_training: carry on with the model called `name`.
    The model file is rewritten after every epoch; the best validation score and its epoch are recorded in it, and the
    weights of that best epoch are kept as <name>_best.pt when a later epoch scored lower.
    """
    import torch
    from .network import FCDenseNet103

    reporter = reporter or Reporter()
    name = name or f'{os.path.basename(os.path.normpath(folder))}_{datetime.date.today():%Y%m%d}'
    out = os.path.join(models_dir(), name + '.pt')
    if continue_training:
        if not os.path.isfile(out):
            raise ValueError(f'There is no model called {name} to continue.')
        base_model = out
    elif os.path.isfile(out):
        raise ValueError(f'A model called {name} already exists. Choose another name, or continue its training.')
    fresh = base_model is None

    train_files, val_files = list_stacks(folder)
    if not train_files:
        raise ValueError('Training needs at least two stacks: one is always kept for validation.')
    device = pick_device(device)
    reporter.stage('training', total=None, message=f'reading {len(train_files) + len(val_files)} stacks from {os.path.basename(os.path.normpath(folder))}')
    images, masks = load_tiles(train_files)
    val_images, val_masks = load_tiles(val_files)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    images, masks = images[order], masks[order]
    picked = random.Random(16).sample(range(len(val_images)), min(VALIDATION_TILES, len(val_images)))
    val_images, val_masks = val_images[picked], val_masks[picked]
    weights = class_weights(masks)

    earlier = 0  # epochs this model had before, when its training is continued
    if fresh:
        torch.manual_seed(seed)
        model, parent = FCDenseNet103(), None
        glorot_init(model)
        model = model.eval().to(device)
    else:
        model, parent_meta = load_model(base_model, device)
        parent = parent_meta.get('base_model') if continue_training else parent_meta.get('name')
        earlier = int(parent_meta.get('epochs') or 0) if continue_training else 0
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

    iterations = len(images) // batch_size
    if iterations == 0:
        raise ValueError(f'Only {len(images)} training tiles: fewer than one batch of {batch_size}.')
    total = epochs * iterations
    meta = {'classes': ['Cell', 'Background'], 'patch_xy': TILE, 'feature_channels': 64, 'framework': 'pytorch',
            'trained_on': f'{datetime.date.today():%Y-%m-%d}', 'mode': 'continued' if continue_training else 'new' if fresh else 'fine-tuned',
            'base_model': parent, 'class_weights': [float(w) for w in weights], 'z': int(images.shape[3]),
            'training_files': [os.path.basename(f) for f in train_files], 'validation_files': [os.path.basename(f) for f in val_files],
            'tiles_train': int(len(images)), 'tiles_val': int(len(val_images)), 'batch_size': batch_size, 'seed': seed}
    reporter.stage('training', total=total, message=f'{len(images)} training tiles, {len(val_images)} validation tiles, '
                                                    f'{epochs} epochs on {device.type}; class weights {weights[0]:.4f} / {weights[1]:.4f}')
    log_path = os.path.join(models_dir(), name + '_training.csv')
    os.makedirs(models_dir(), exist_ok=True)
    with open(log_path, 'a' if continue_training and os.path.isfile(log_path) else 'w', newline='') as f:
        if f.tell() == 0:
            csv.writer(f).writerow(['epoch', 'learning_rate', 'loss', 'validation_accuracy', 'validation_iou', 'seconds'])

    best_iou, best_epoch, started = -1.0, 0, time.time()
    for epoch in range(epochs):
        rate = learning_rate(epoch, fresh)
        for group in optimizer.param_groups:
            group['lr'] = rate
        losses, epoch_start = [], time.time()
        for i in range(iterations):
            x = torch.from_numpy(images[i * batch_size:(i + 1) * batch_size])[:, None].to(device)
            target = torch.from_numpy((~masks[i * batch_size:(i + 1) * batch_size]).astype(np.int64)).to(device)
            optimizer.zero_grad()
            loss = loss_function(model(x)[0], target, weight_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % 5 == 0 or i + 1 == iterations:
                reporter.progress(epoch * iterations + i + 1, total, f'epoch {epoch + 1} of {epochs}, loss {np.mean(losses[-20:]):.4f}')
        accuracy, iou = validate(model, device, val_images, val_masks)
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise RuntimeError(f'The loss became {mean_loss} in epoch {epoch + 1}; training stopped.')
        improved = iou > best_iou
        if improved:
            best_iou, best_epoch = iou, epoch
        meta.update(epochs=earlier + epoch + 1, best_val_iou=best_iou, best_epoch=earlier + best_epoch + 1, final_val_iou=iou,
                    final_loss=mean_loss)
        save_model(out, model, meta)
        best_path = os.path.join(models_dir(), name + '_best.pt')
        if improved and epoch > 0:
            save_model(best_path, model, dict(meta, note=f'best validation epoch of {name}'))
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([earlier + epoch + 1, rate, round(mean_loss, 6), round(accuracy, 6), round(iou, 6), round(time.time() - epoch_start, 1)])
        reporter.epoch(epoch=epoch + 1, epochs=epochs, loss=mean_loss, validation_iou=iou, validation_accuracy=accuracy,
                       best_iou=best_iou, best_epoch=best_epoch + 1, learning_rate=rate)
    if best_epoch == epochs - 1:  # the last epoch is the best one: the separate copy is not needed
        for ext in ('.pt', '.json'):
            if os.path.isfile(os.path.join(models_dir(), name + '_best' + ext)):
                os.remove(os.path.join(models_dir(), name + '_best' + ext))
    result = {'model': out, 'name': name, 'epochs': epochs, 'best_val_iou': best_iou, 'best_epoch': best_epoch + 1,
              'final_val_iou': meta['final_val_iou'], 'log': log_path, 'seconds': round(time.time() - started, 1)}
    reporter.stage_done(**result)
    return result
