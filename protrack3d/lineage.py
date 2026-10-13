"""The lineage of a tracked movie, read from the tables tracking writes. Light enough to load inside the app.

Families follow FamilyTreeComplete.findFamilies: objects connected through splits and merges, in both directions.
"""
import csv
import os
from collections import Counter


def _rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as f:
        return [{k: int(float(v)) for k, v in row.items() if v not in ('', None)} for row in csv.DictReader(f)]


class Lineage:
    def __init__(self, track_folder, partial_merges=False):
        self.folder = track_folder
        self.objects = {r['index']: (r['timestart'], r['timeend'], r['parent']) for r in _rows(os.path.join(track_folder, 'Sheet3.csv'))}
        if not self.objects:
            raise ValueError('This folder has no Sheet3.csv: choose a tracking results folder.')
        seen, self.splits = set(), []
        for r in _rows(os.path.join(track_folder, 'split_list.csv')):  # (parent, child, time)
            if r['Splitted Into'] not in seen:
                seen.add(r['Splitted Into'])
                self.splits.append((r['Splitted'], r['Splitted Into'], r['Time']))
        self.merges = [(r['Merged'], r['Merged Into'], r['Time']) for r in _rows(os.path.join(track_folder, 'merge_list.csv'))]
        self.partial = [(r['Merged'], r['Merged Into'], r['Time']) for r in _rows(os.path.join(track_folder, 'split-and-merge_list.csv'))]
        self.partial_merges = partial_merges
        self._group()
        self.first = min(s for s, _, _ in self.objects.values())
        self.last = max(e for _, e, _ in self.objects.values())

    def _group(self):
        root = {i: i for i in self.objects}

        def find(a):
            while root[a] != a:
                root[a] = root[root[a]]
                a = root[a]
            return a

        links = [(p, i) for i, (_, _, p) in self.objects.items()] + [(a, b) for a, b, _ in self.splits + self.merges]
        if self.partial_merges:
            links += [(a, b) for a, b, _ in self.partial]
        for a, b in links:
            if a in root and b in root:
                a, b = find(a), find(b)
                root[max(a, b)] = min(a, b)
        self.family = {i: find(i) for i in root}
        self.family_size = Counter(self.family.values())

    def members(self, obj):
        fam = self.family[obj]
        return sorted(i for i, f in self.family.items() if f == fam)

    def order(self, min_lifetime=1):
        """Objects in drawing order: families of two or more first (largest first, then by first appearance), then
        the unrelated objects by first appearance. Unrelated objects shorter than min_lifetime are left out."""
        born = {}
        for i, (start, _, _) in self.objects.items():
            f = self.family[i]
            born[f] = min(born.get(f, start), start)
        related = [i for i in self.objects if self.family_size[self.family[i]] > 1]
        alone = [i for i in self.objects if self.family_size[self.family[i]] == 1
                 and self.objects[i][1] - self.objects[i][0] + 1 >= min_lifetime]
        related.sort(key=lambda i: (-self.family_size[self.family[i]], born[self.family[i]], self.family[i], self.objects[i][0], i))
        alone.sort(key=lambda i: (self.objects[i][0], i))
        return related + alone

    def describe(self, obj):
        start, end, parent = self.objects[obj]
        n = self.family_size[self.family[obj]]
        parts = [f'Object {obj}', f'timepoints {start}–{end}' if end > start else f'timepoint {start}',
                 f'family {self.family[obj]} ({n} objects)' if n > 1 else 'no relatives']
        events = [f'split from {a} at {t}' for a, b, t in self.splits if b == obj]
        events += [f'split off {b} at {t}' for a, b, t in self.splits if a == obj]
        events += [f'merged into {b} at {t}' for a, b, t in self.merges if a == obj]
        events += [f'absorbed {a} at {t}' for a, b, t in self.merges if b == obj]
        return '   ·   '.join(parts + events[:6] + (['…'] if len(events) > 6 else []))

    def image_path(self):
        """The image this tracking belongs to, when it sits next to the results under its usual name."""
        folder = os.path.normpath(self.folder)
        base = os.path.basename(folder)
        if base.endswith('_TrackingOutput'):
            for ext in ('.tif', '.tiff', '.nii'):
                path = os.path.join(os.path.dirname(folder), base[:-len('_TrackingOutput')] + ext)
                if os.path.isfile(path):
                    return path
        return None
