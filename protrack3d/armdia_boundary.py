"""Boundary analysis for the Armadillo workflow: which puncta sit on a cell boundary, and how bright they are.

A port of the notebook "Boundary Dialate and ARM quantify.ipynb", with its two steps:

  1. every 2D cell-outline image is thickened from `current_width` to `target_width` pixels, repeated on every z
     slice, and saved as <image>_Boundary_<width>.tif under <outlines>/Dilated/Embryo NN/;
  2. every punctum of <image>_ARM_SEG.tif is called a boundary punctum when at least `overlap` of its voxels lie on
     the thickened outline; per-punctum and per-image tables are written.

The notebook measures Armadillo only. When <image>_DIA.tif is there too, Dia is measured in the same puncta and its
columns (ending in _dia) are added after the Armadillo ones; the Armadillo columns are unchanged.

Folders are matched by embryo number ("embryo 3", "Embryo 03") and files by name. An outline file is the image name
with, optionally, a channel or projection prefix in front (C2-, C1-MAX_, MAX_C2-, MAX_) and an outline word at the end
(outline only, outlineonly, outline_only, cell outline, outline). Names are compared ignoring case and treating spaces,
underscores and hyphens alike. Outlines may be .tif or .tiff, or an ImageJ TIFF saved under a .czi name. Files with the
outline drawn into the picture ("with outline", "withoutline", even misspelt "withoutilne") are not outlines and are skipped.
"""
import os
import re
from pathlib import Path

import numpy as np

from .events import Reporter

BOUNDARY_VALUE = 60000
INTERIOR_VALUE = 20000


def extract_embryo_key(name):
    m = re.search(r'embryo\s*0*(\d+)', name, re.IGNORECASE)
    return int(m.group(1)) if m else None


PREFIX = re.compile(r'^(?:MAX_?)?(?:C\d+-)?(?:MAX_?)?', re.IGNORECASE)  # C2-, C1-MAX_, MAX_C2-, MAX_
OUTLINE_WORD = re.compile(r'[-_ ]*(?:cell[-_ ]*)?outl\w*?(?:[-_ ]*only)?\s*$', re.IGNORECASE)  # outline only, outlineonly, _outline, cell outline
WITH_OUTLINE = re.compile(r'with[-_ ]*out', re.IGNORECASE)  # with outline, withoutline, withoutilne: the picture with lines drawn in
OUTLINE_EXTENSIONS = ('.tif', '.tiff', '.czi')


def match_key(name):
    """What two names are compared by: lower case, with every run of spaces, underscores and hyphens made one space."""
    return re.sub(r'[-_\s]+', ' ', name).strip().lower()


def strip_boundary_filename(fn):
    stem = Path(fn).stem
    stem = PREFIX.sub('', stem)
    stem = OUTLINE_WORD.sub('', stem)
    return stem.strip('-_ ')


def read_outline(path):
    """A 2-D outline image. ImageJ sometimes saves a TIFF under a .czi name, so the file's content decides, not its name."""
    import tifffile
    try:
        data = tifffile.imread(path)
    except Exception:
        if str(path).lower().endswith('.czi'):
            from .czi import read_czi
            data = read_czi(str(path))[0]
        else:
            raise
    data = np.squeeze(np.asarray(data))
    if data.ndim != 2:
        raise ValueError(f'{Path(path).name} is not a single 2-D outline (its shape is {data.shape})')
    return data


def embryo_folders(root, skip=()):
    found = {}
    for entry in Path(root).iterdir():
        if entry.is_dir() and entry.name.lower() not in skip:
            key = extract_embryo_key(entry.name)
            if key is not None:
                found[key] = entry
    return found


def arm_images(folder, z_suffix):
    """{name used for matching: (full name, path)} for every <name>_ARM.tif in an embryo folder."""
    out = {}
    for p in folder.glob('*_ARM.tif'):
        full = p.stem[:-len('_ARM')]
        key = full[:-len(z_suffix)] if z_suffix and full.endswith(z_suffix) else full
        out[match_key(key)] = (full, p)
    return out


def outline_files(folder):
    """{match key: outline path}. A .tif wins over a .czi-named copy of the same outline."""
    out = {}
    for fn in sorted(os.listdir(folder), key=lambda f: f.lower().endswith('.czi')):
        if fn.startswith('.') or not fn.lower().endswith(OUTLINE_EXTENSIONS) or WITH_OUTLINE.search(fn):
            continue
        stem = Path(fn).stem
        if not OUTLINE_WORD.search(stem) and not PREFIX.match(stem).group(0):
            continue  # a .czi or .tif with no outline word and no channel prefix is an image, not an outline
        out.setdefault(match_key(strip_boundary_filename(fn)), folder / fn)
    return out


def dilate_outline(outline, current_width, target_width):
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk
    iterations = int(np.ceil((target_width - current_width) / 2))
    return binary_dilation(np.asarray(outline) > 0, structure=disk(1), iterations=iterations), iterations


def analyze_image(name, image_folder, boundary_path, embryo, root, overlap):
    import pandas as pd
    import tifffile
    from skimage.measure import label, regionprops
    arm_path = image_folder / f'{name}_ARM.tif'
    seg_path = image_folder / f'{name}_ARM_SEG.tif'
    if not seg_path.exists():
        raise FileNotFoundError(f'no segmentation for {arm_path.name}: run the Armadillo workflow first')
    arm = tifffile.imread(arm_path).astype(np.float64)
    puncta = np.asarray(tifffile.imread(seg_path))
    boundary = np.asarray(tifffile.imread(boundary_path)) > 0
    if puncta.shape != arm.shape or boundary.shape != arm.shape:
        raise ValueError(f'shapes differ: image {arm.shape}, mask {puncta.shape}, boundary {boundary.shape}')

    dia_path = image_folder / f'{name}_DIA.tif'
    dia = tifffile.imread(dia_path).astype(np.float64) if dia_path.exists() else None
    if dia is not None and dia.shape != arm.shape:
        raise ValueError(f'shapes differ: Armadillo image {arm.shape}, Dia image {dia.shape}')

    labels = label(puncta > 0, connectivity=1)
    props = regionprops(labels, intensity_image=arm)
    rows = []
    for p in props:
        coords = p.coords
        on = boundary[coords[:, 0], coords[:, 1], coords[:, 2]].sum() / coords.shape[0]
        total = p.intensity_image[p.image].sum()
        rows.append(dict(label=p.label, voxel_count=coords.shape[0], total_intensity=total,
                         mean_intensity=total / coords.shape[0], overlap_frac=on, on_boundary=on >= overlap))
    df = pd.DataFrame(rows)
    rel_path = str(arm_path.relative_to(Path(root).parent))
    df['imagename'], df['embryo'], df['threshold'] = rel_path, embryo, overlap
    if dia is not None:
        df['total_intensity_dia'] = [dia[p.coords[:, 0], p.coords[:, 1], p.coords[:, 2]].sum() for p in props]
        df['mean_intensity_dia'] = df['total_intensity_dia'] / df['voxel_count']
    pct = int(overlap * 100)
    df.to_csv(image_folder / f'{name}_puncta_table_{pct}pct.csv', index=False)

    out = np.zeros(arm.shape, dtype=np.uint16)
    for row in df.itertuples():
        coords = props[row.Index].coords
        out[coords[:, 0], coords[:, 1], coords[:, 2]] = BOUNDARY_VALUE if row.on_boundary else INTERIOR_VALUE
    tifffile.imwrite(image_folder / f'{name}_Boundary_{pct}pct.tif', out)

    b, i = df[df.on_boundary], df[~df.on_boundary]
    b_vox, i_vox = b.voxel_count.sum(), i.voxel_count.sum()
    background = labels == 0
    bg_vox, bg_int = background.sum(), arm[background].sum()
    summary = dict(
        imagename=rel_path, embryo=embryo, threshold=overlap,
        n_puncta_boundary=len(b), total_intensity_boundary=b.total_intensity.sum(),
        mean_per_punctum_intensity_boundary=b.total_intensity.mean() if len(b) else np.nan,
        median_per_punctum_intensity_boundary=b.total_intensity.median() if len(b) else np.nan,
        boundary_intensity_density=b.total_intensity.sum() / b_vox if b_vox else np.nan,
        n_puncta_interior=len(i), total_intensity_interior=i.total_intensity.sum(),
        mean_per_punctum_intensity_interior=i.total_intensity.mean() if len(i) else np.nan,
        median_per_punctum_intensity_interior=i.total_intensity.median() if len(i) else np.nan,
        interior_intensity_density=i.total_intensity.sum() / i_vox if i_vox else np.nan,
        background_voxels=bg_vox, background_intensity=bg_int,
        background_intensity_density=bg_int / bg_vox if bg_vox else np.nan)
    if dia is not None:
        bg_dia = dia[background].sum()
        summary.update(
            total_intensity_boundary_dia=b.total_intensity_dia.sum(),
            mean_per_punctum_intensity_boundary_dia=b.total_intensity_dia.mean() if len(b) else np.nan,
            median_per_punctum_intensity_boundary_dia=b.total_intensity_dia.median() if len(b) else np.nan,
            boundary_intensity_density_dia=b.total_intensity_dia.sum() / b_vox if b_vox else np.nan,
            total_intensity_interior_dia=i.total_intensity_dia.sum(),
            mean_per_punctum_intensity_interior_dia=i.total_intensity_dia.mean() if len(i) else np.nan,
            median_per_punctum_intensity_interior_dia=i.total_intensity_dia.median() if len(i) else np.nan,
            interior_intensity_density_dia=i.total_intensity_dia.sum() / i_vox if i_vox else np.nan,
            background_intensity_dia=bg_dia, background_intensity_density_dia=bg_dia / bg_vox if bg_vox else np.nan)
    return df, summary


def run(root, embryos=None, outlines=None, current_width=6, target_width=30, overlap=0.30, z_suffix='-20z', reporter=None):
    """Run both steps. `root` holds the embryo folders (directly, or in "Embryos"); the "Cell Outlines" folder is
    inside it or next to it, unless `outlines` names another."""
    import pandas as pd
    import tifffile
    reporter = reporter or Reporter()
    root = Path(root)
    embryos_root = Path(embryos) if embryos else (root / 'Embryos' if (root / 'Embryos').is_dir() else root)
    outlines_root = Path(outlines) if outlines else root / 'Cell Outlines'
    if not outlines and not outlines_root.is_dir() and (root.parent / 'Cell Outlines').is_dir():
        outlines_root = root.parent / 'Cell Outlines'  # kept next to the folder of embryos rather than inside it
    if not outlines_root.is_dir():
        raise ValueError(f'There is no outlines folder at {outlines_root}.')
    image_folders = embryo_folders(embryos_root, skip=('cell outlines', 'dilated'))
    outline_folders = embryo_folders(outlines_root, skip=('dilated',))
    notes = []
    for key in sorted(set(image_folders) - set(outline_folders)):
        notes.append(f'embryo {key}: no outline folder')
    for key in sorted(set(outline_folders) - set(image_folders)):
        notes.append(f'embryo {key}: outlines but no image folder')

    pairs = []  # (embryo, image folder, full name, outline path)
    for key in sorted(set(image_folders) & set(outline_folders)):
        images, lines = arm_images(image_folders[key], z_suffix), outline_files(outline_folders[key])
        for name in sorted(set(images) - set(lines)):
            notes.append(f'embryo {key}: no outline for {name}')
        for name in sorted(set(lines) - set(images)):
            notes.append(f'embryo {key}: outline {name} matches no image')
        pairs += [(key, image_folders[key], images[m][0], images[m][1], lines[m]) for m in sorted(set(images) & set(lines))]
    if not pairs:
        raise ValueError('No image could be matched to an outline. ' + '; '.join(notes[:6]))

    pct = int(overlap * 100)
    reporter.stage('boundary analysis', total=len(pairs), message=f'{len(pairs)} images; outlines thickened from {current_width} to {target_width} pixels; '
                                                                  f'a punctum is on the boundary when {pct} % of it overlaps')
    tables, summaries = [], []
    for n, (key, folder, name, arm_path, outline_path) in enumerate(pairs):
        try:
            z, y, x = tifffile.imread(arm_path).shape
            outline = read_outline(outline_path)
            if outline.shape != (y, x):
                raise ValueError(f'outline is {outline.shape}, image is {(y, x)}')
            dilated, _ = dilate_outline(outline, current_width, target_width)
            out_folder = outlines_root / 'Dilated' / f'Embryo {key:02d}'
            out_folder.mkdir(parents=True, exist_ok=True)
            boundary_path = out_folder / f'{name}_Boundary_{target_width}.tif'
            tifffile.imwrite(boundary_path, (np.broadcast_to(dilated, (z, y, x)).astype(np.uint8) * 65535).astype(np.uint16))
            df, summary = analyze_image(name, folder, boundary_path, key, root, overlap)
            tables.append(df)
            summaries.append(summary)
            reporter.progress(n + 1, len(pairs), f'{name}: {summary["n_puncta_boundary"]} on the boundary, {summary["n_puncta_interior"]} inside')
        except Exception as e:
            notes.append(f'embryo {key}, {name}: {e}')
            reporter.progress(n + 1, len(pairs), f'{name}: skipped ({e})')
    result = {'images': len(summaries), 'skipped': notes}
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(root / f'all_puncta_{pct}pct.csv', index=False)
        result['table'] = str(root / f'all_threshold_summary_{pct}pct_wide.csv')
        pd.DataFrame(summaries).to_csv(result['table'], index=False)
    reporter.stage_done(**result)
    return result
