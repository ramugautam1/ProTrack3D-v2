"""Complete family trees, and one overview of every object.

generateFamilyTree (FamilyTreeWithSplitAndMerge.py) starts from one object and follows its future. The functions here
follow the past as well: a complete family is the object, its ancestors, everything that merged with it or with any
relative, and all of their descendants, until nothing more is connected. Each family is drawn once.

Inputs are the tables tracking writes: Sheet3.csv (index, timestart, timeend, parent), split_list.csv, merge_list.csv
and split-and-merge_list.csv.
"""
import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from FamilyTreeWithSplitAndMerge import familyExtent, keep_values, objectBoxes, range_to_list

COLORS = ["coral", "blue", "brown", "chartreuse", "aquamarine", "cyan", "darkorange", "darkred", "dodgerblue",
          "firebrick", "forestgreen", "fuchsia", "gold", "green", "hotpink", "indigo", "lime", "magenta", "maroon",
          "mediumblue", "mediumspringgreen", "navy", "olive", "orange", "orangered", "orchid", "peru", "purple", "red",
          "royalblue", "saddlebrown", "seagreen", "sienna", "skyblue", "springgreen", "teal", "tomato", "turquoise",
          "violet", "yellow", "yellowgreen", "burlywood", "cadetblue", "cornflowerblue", "darkcyan", "darkgoldenrod",
          "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorchid", "darksalmon", "darkseagreen",
          "darkslateblue", "darkslategray", "darkturquoise", "darkviolet", "deepskyblue"]


def readTracking(trackFolder):
    """The tracking tables as (objects, splits, merges, partial merges)."""
    objects = pd.read_csv(trackFolder + '/Sheet3.csv')
    splits = pd.read_csv(trackFolder + '/split_list.csv')
    splits = splits[~splits['Splitted Into'].duplicated(keep='first')]
    merges = pd.read_csv(trackFolder + '/merge_list.csv')
    partial = trackFolder + '/split-and-merge_list.csv'
    partial = pd.read_csv(partial) if os.path.isfile(partial) else merges.iloc[0:0]
    return objects, splits, merges, partial


def findFamilies(objects, splits, merges, partial=None):
    """Group the objects into complete families. Returns {object id: family id}; a family id is its smallest member."""
    root = {int(i): int(i) for i in objects['index']}

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    links = list(zip(objects['parent'], objects['index'])) + list(zip(splits['Splitted'], splits['Splitted Into']))
    links += list(zip(merges['Merged'], merges['Merged Into']))
    if partial is not None:
        links += list(zip(partial['Merged'], partial['Merged Into']))
    for a, b in links:
        if a in root and b in root:
            a, b = find(int(a)), find(int(b))
            root[max(a, b)] = min(a, b)
    return {i: find(i) for i in root}


LARGE_FAMILY = 40        # families with more objects than this are drawn compactly, and saved as a PDF as well
MAX_3D_OBJECTS = 30      # the 3D panels are skipped for families larger than this ...
MAX_3D_TIMEPOINTS = 60   # ... or longer than this: one panel per timepoint stops being readable, and is very slow


def drawTreeDiagram(members, splits, merges, partial, filename, title=None):
    """Lineage diagram of one family: one row per object, a dot per timepoint it exists, splits in blue, merges in red."""
    members = members.reset_index(drop=True)
    row = {int(j): i + 1 for i, j in enumerate(members['index'])}
    first, last = int(members.timestart.min()), int(members.timeend.max())
    compact = len(members) > LARGE_FAMILY
    if compact:
        per_row = min(0.2, 130 / len(members))  # inches; the picture stays under 130 inches tall
        scale = per_row / 1.0
        figsize, dot, font, tick, line = ((last - first + 4) * 0.25 + 2, len(members) * per_row + 2), max(1, 200 * scale ** 2), max(3, 20 * scale * 1.6), 7, 0.6
    else:
        figsize, dot, font, tick, line = (last - first + 4, len(members) + 2), 200, 20, 15, 1.5
    fig = plt.figure(num=1, clear=True, figsize=figsize)
    ax = plt.subplot()
    ax.set_ylim(0, len(members) + 2)
    ax.set_xlim(first - 1, last + 2)
    ax.set_xlabel('timepoints (t)', fontsize=font if not compact else 10, color='k')
    ax.set_xticks(np.arange(first, last + 1, 1 if not compact else max(1, (last - first) // 40)))
    ax.tick_params(axis='x', labelsize=tick)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=font if not compact else 12)
    for j, start, end in zip(members['index'], members['timestart'], members['timeend']):
        y = row[int(j)]
        ax.plot([start, end], [y, y], color='green', linewidth=line)
        ax.scatter(range(int(start), int(end) + 1), [y] * (int(end) - int(start) + 1), color=COLORS[int(j) % len(COLORS)], s=dot, zorder=3)
        ax.text(end + (0.3 if compact else 0), y + (0 if compact else 0.25), str(int(j)), fontsize=font, va='center' if compact else 'baseline')
    for table, source, target, style in ((splits, 'Splitted', 'Splitted Into', dict(color='darkblue')),
                                         (merges, 'Merged', 'Merged Into', dict(color='red', linestyle='--')),
                                         (partial, 'Merged', 'Merged Into', dict(color='darkorange', linestyle=':'))):
        if table is None:
            continue
        for a, b, t in zip(table[source], table[target], table['Time']):
            if a in row and b in row:
                ax.plot([t - 1, t], [row[a], row[b]], linewidth=line, **style)
    plt.savefig(filename, bbox_inches='tight' if compact else None)
    if compact:
        plt.savefig(os.path.splitext(filename)[0] + '.pdf', bbox_inches='tight')
    plt.close()


def draw3D(tracked, boxes, ids, first, last, filename, title):
    """The family in 3D, one panel per timepoint, as generateFamilyTree draws it."""
    extent = familyExtent(boxes, ids)
    total = last - first + 3
    rows = int(np.ceil(total / 5))
    fig = plt.figure(num=1, clear=True, figsize=(20, rows * 4), constrained_layout=True)
    fig.patch.set_facecolor('white')
    fig.suptitle(title)
    for i in range(max(first - 2, 0), min(last + 1, tracked.shape[-1])):
        volume = keep_values(np.asarray(tracked.dataobj[..., i]).astype(np.float32), ids)
        ax = fig.add_subplot(rows, 5, i - first + 3, projection='3d')
        ax.xaxis.set_tick_params(labelbottom=True, labelsize=12)
        ax.yaxis.set_tick_params(labelleft=True, labelsize=12)
        ax.zaxis.set_ticklabels([])
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        present = [int(v) for v in np.unique(volume) if v]
        ax.set_title(f't={i + 1}  {present}')
        ax.set_box_aspect([1, 1, 0.25])
        for obj in ids:
            x, y, z = np.where(volume == obj)
            if len(x) > 2:
                try:
                    if len(np.unique(z)) == 1:
                        z = [zz + 1 if n % 2 == 0 else zz for n, zz in enumerate(z)]
                    ax.plot_trisurf(x, y, z, color=COLORS[int(obj) % len(COLORS)], alpha=0.6)
                    ax.text(x=x[0], y=y[0], z=z[0], s=str(int(obj)), fontsize=12)
                except Exception:
                    pass
        ax.set_xlim(extent[0][0] - 4, extent[0][1] + 4)
        ax.set_ylim(extent[1][0] - 4, extent[1][1] + 4)
        ax.set_zlim(extent[2][0] - 1, extent[2][1] + 1)
        ax.view_init(40, 50)
    plt.savefig(filename)
    plt.close()


def generateCompleteFamilyTree(excelFile, ftFolder, tidlist, partialMerges=False, with3D=True):
    """Draw the complete family of every requested object.

    excelFile: any file in the tracking results folder, as for generateFamilyTree.
    tidlist: '76', '1,2,13', '1-29', or 'all' (every family of two or more objects).
    partialMerges: also connect objects through split-and-merge events (a piece of one object joining another).
    Writes TreeDiagrams_Complete/, FamilyTrees_3D_Complete/ and csvFiles_Complete/ into ftFolder, plus families.csv,
    which lists every object with the family it belongs to. Large families are drawn compactly (with a PDF that can be
    zoomed), and the 3D panels are skipped for very large or very long ones.
    """
    trackFolder = os.path.dirname(excelFile)
    objects, splits, merges, partial = readTracking(trackFolder)
    partial = partial if partialMerges else None
    family = findFamilies(objects, splits, merges, partial)
    size = pd.Series(family).value_counts()

    folders = {name: os.path.join(ftFolder, name) for name in ('TreeDiagrams_Complete', 'FamilyTrees_3D_Complete', 'csvFiles_Complete')}
    for path in folders.values():
        os.makedirs(path, exist_ok=True)
    table = objects.assign(family=objects['index'].map(family))
    table['family size'] = table['family'].map(size)
    table.to_csv(os.path.join(ftFolder, 'families.csv'), index=False)

    if re.sub(r'[^a-zA-Z0-9 ]', '', tidlist).lower() == 'all':
        wanted = sorted(f for f in size.index if size[f] > 1)
    else:
        wanted = list(dict.fromkeys(family[i] for i in range_to_list(tidlist) if i in family))
    print('\n=================================================\n')
    print(f'Drawing {len(wanted)} complete families')
    if with3D and wanted:
        tracked = nib.load(trackFolder + '/TrackedCombined.nii')
        boxes = objectBoxes(tracked)
    for fam in wanted:
        print(f'\rProcessing ID {fam:05d}', end='', flush=True)
        members = table[table['family'] == fam]
        ids = [int(i) for i in members['index']]
        members.to_csv(os.path.join(folders['csvFiles_Complete'], f'family_{fam}.csv'), index=False)
        name = f'FT_complete_ID_{fam:05d}'
        title = f'Family {fam}: {len(ids)} object' + ('s' if len(ids) != 1 else '')
        drawTreeDiagram(members, splits, merges, partial, os.path.join(folders['TreeDiagrams_Complete'], name + '.png'), title)
        span = int(members.timeend.max() - members.timestart.min()) + 1
        if with3D and (len(ids) > MAX_3D_OBJECTS or span > MAX_3D_TIMEPOINTS):
            print(f'\nFamily {fam} has {len(ids)} objects over {span} timepoints: 3D panels skipped.')
        elif with3D:
            draw3D(tracked, boxes, ids, int(members.timestart.min()), int(members.timeend.max()),
                   os.path.join(folders['FamilyTrees_3D_Complete'], name + '.png'), title)
    print()
    return wanted


NODE_ROW = 0.07  # inches per object below which dots would run together, so objects are drawn as solid lines


def drawAllObjects(excelFile, filename, partialMerges=False, minLifetime=1, rowHeight=0.1, timeWidth=0.2,
                   maxHeight=130):
    """One picture of the whole movie, a row per object, drawn like the family-tree diagrams: a dot for every
    timepoint an object exists, joined by its life line, with splits in blue and merges in dashed red.

    Families of two or more objects come first, each in its own colour with its family id on the left; objects with no
    relatives follow in grey. minLifetime leaves out unrelated objects that exist for fewer timepoints.
    rowHeight and timeWidth are inches per object and per timepoint. The picture is kept under maxHeight inches
    (19,500 pixels) by making rows thinner; with more than about 1,850 objects there is no room for dots, and each
    object becomes a solid line with ticks for its events. A PDF is saved next to the PNG: it can be zoomed until
    every id is legible.
    """
    trackFolder = os.path.dirname(excelFile)
    objects, splits, merges, partial = readTracking(trackFolder)
    partial = partial if partialMerges else None
    family = findFamilies(objects, splits, merges, partial)
    table = objects.assign(family=objects['index'].map(family))
    table['size'] = table['family'].map(table['family'].value_counts())
    table['life'] = table['timeend'] - table['timestart'] + 1
    table = table[(table['size'] > 1) | (table['life'] >= minLifetime)]
    table['born'] = table['family'].map(table.groupby('family')['timestart'].min())
    table = table.sort_values(['size', 'born', 'family', 'timestart', 'index'], ascending=[False, True, True, True, True])
    table = pd.concat([table[table['size'] > 1], table[table['size'] == 1].sort_values(['timestart', 'index'])]).reset_index(drop=True)
    row = {int(j): i for i, j in enumerate(table['index'])}
    first, last = int(table.timestart.min()), int(table.timeend.max())
    rowHeight = min(rowHeight, maxHeight / max(1, len(table)))

    fig, ax = plt.subplots(figsize=(max(6, (last - first + 3) * timeWidth + 1.2), max(4, len(table) * rowHeight + 1.2)))
    related = (table['size'] > 1).to_numpy()
    colour = [COLORS[int(f) % len(COLORS)] if r else '0.6' for f, r in zip(table['family'], related)]
    y = np.arange(len(table))
    nodes = rowHeight >= NODE_ROW
    if nodes:
        ax.add_collection(LineCollection([[(s, i), (e, i)] for s, e, i in zip(table.timestart, table.timeend, y)], colors=colour, linewidths=0.8))
        xs = np.concatenate([np.arange(s, e + 1) for s, e in zip(table.timestart, table.timeend)])
        ys = np.concatenate([np.full(e - s + 1, i) for s, e, i in zip(table.timestart, table.timeend, y)])
        cs = [c for c, s, e in zip(colour, table.timestart, table.timeend) for _ in range(e - s + 1)]
        ax.scatter(xs, ys, s=(min(rowHeight, timeWidth) * 72 * 0.62) ** 2, c=cs, linewidths=0, zorder=3)
    else:  # half a timepoint on each side, so an object that exists for one timepoint is still visible
        ax.add_collection(LineCollection([[(s - 0.4, i), (e + 0.4, i)] for s, e, i in zip(table.timestart, table.timeend, y)],
                                         colors=colour, linewidths=max(0.6, rowHeight * 72 * 0.55)))
    for links, source, target, style in ((splits, 'Splitted', 'Splitted Into', dict(colors='darkblue')),
                                         (merges, 'Merged', 'Merged Into', dict(colors='red', linestyles='--' if nodes else '-')),
                                         (partial, 'Merged', 'Merged Into', dict(colors='darkorange', linestyles=':'))):
        if links is None:
            continue
        pairs = [(row[a], row[b], t) for a, b, t in zip(links[source], links[target], links['Time']) if a in row and b in row]
        # with dots: from the first object one timepoint earlier to the other object now, as in the family trees
        segments = [[(t - 1, ra), (t, rb)] if nodes else [(t - 0.5, ra), (t - 0.5, rb)] for ra, rb, t in pairs]
        ax.add_collection(LineCollection(segments, linewidths=0.8, **style))
    blocks = table[related].groupby('family', sort=False).apply(lambda g: (g.index.min(), g.index.max()))
    for fam, (top, bottom) in blocks.items():
        ax.text(first - 0.8, (top + bottom) / 2, str(int(fam)), ha='right', va='center', fontsize=max(3, min(7, rowHeight * 72 * (bottom - top + 1) * 0.8)))
    ax.set_xlim(first - 2.5, last + 1)
    ax.set_ylim(len(table), -1)
    ax.set_yticks([])
    ax.set_xlabel('timepoint')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labeltop=True)
    n_fam = int(blocks.shape[0])
    ax.set_title(f'{len(table)} objects: {int(related.sum())} in {n_fam} families, {int((~related).sum())} without relatives', fontsize=9, pad=18)
    for side in ('left', 'right'):
        ax.spines[side].set_visible(False)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(filename)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return {'objects': int(len(table)), 'families': n_fam, 'width_px': int(fig.get_figwidth() * 150), 'height_px': int(fig.get_figheight() * 150)}
