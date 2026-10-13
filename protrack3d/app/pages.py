"""The app's pages. Each page collects inputs and submits a job; the Jobs page shows what is running."""
import csv
import os

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import batch, stages
from ..models import list_models
from ..paths import models_dir
from ..segmentation import read_image_info, segmentation_folder
from . import icons
from .jobs import STAGE_LABELS, Job, format_seconds
from .widgets import Card, ModelBox, NumberField, PathField, SmoothBar, button, text

IMAGE_FILTERS = 'Images (*.tif *.tiff *.nii);;All files (*)'
STEP_STAGE = {'seg': 'segmentation', 'track': 'tracking', 'analysis': 'analysis'}


def describe(info):
    line = f"{info['x']} × {info['y']} × {info['z']} voxels  ·  {info['t']} timepoint{'s' if info['t'] != 1 else ''}"
    return line


def size_text(gb):
    return f'about {gb:.1f} GB' if gb >= 1 else f'about {max(1, round(gb * 1000))} MB'


def disk_estimate(info, timepoints, tracking):
    """Rough size of the outputs: 64 feature channels per voxel, plus tracking's working copies."""
    voxels = info['x'] * info['y'] * info['z']
    per_timepoint = voxels * 64 * 4 + (voxels * (64 * 8 + 12) if tracking else 0)
    return size_text(per_timepoint * timepoints / 1e9)


def segmented_frames(seg_folder):
    try:
        with open(os.path.join(seg_folder, 'segmentation_parameters.csv')) as f:
            row = next(csv.DictReader(f))
        return int(row['endTime']) - int(row['startTime']) + 1
    except (OSError, StopIteration, KeyError, ValueError):
        return None


def spin(low=1, high=99999, value=1, width=90):
    return NumberField(low, high, value, width)


def inline(*widgets):
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    for item in widgets:
        row.addWidget(item)
    row.addStretch(1)
    return w


def rename(widget, name):
    widget.setObjectName(name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class Page(QScrollArea):
    def __init__(self, ctx, title, subtitle):
        super().__init__()
        self.ctx = ctx
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName('page')
        self.setWidget(inner)
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(36, 30, 36, 30)
        self.body.setSpacing(16)
        self.body.addWidget(text(title, 'h1'))
        if subtitle:
            self.body.addWidget(text(subtitle, 'sub', wrap=True))
            self.body.addSpacing(4)

    def add_run_bar(self, label, handler):
        self.error = text('', 'error', wrap=True)
        self.run_button = button(label, 'primary')
        self.run_button.clicked.connect(lambda: self._run(handler))
        bar = QHBoxLayout()
        bar.addWidget(self.error, 1)
        bar.addWidget(self.run_button)
        self.body.addLayout(bar)
        self.body.addStretch(1)

    def _run(self, handler):
        problem = handler()
        self.error.setText(problem or '')


class ImagePage(Page):
    """A page that starts from one image and shows what the image is."""

    def add_image_field(self, card):
        self.info = None
        self.image = PathField(self.ctx.theme, 'file', 'Drop an image here (.tif or .nii), or browse', IMAGE_FILTERS)
        self.image.changed.connect(self._image_changed)
        card.row('Image', self.image)

    def _image_changed(self, path):
        self.info = None
        if path:
            try:
                self.info = read_image_info(path)
                self.image.set_detail(describe(self.info))
            except Exception as e:
                self.image.set_detail(str(e))
        self.image_changed(path)

    def image_changed(self, path):
        pass

    def set_image(self, path):
        self.image.set_path(path)


class SegmentPage(ImagePage):
    def __init__(self, ctx):
        super().__init__(ctx, 'Segmentation', 'Find the objects in every timepoint of a 3D image with a trained model.')
        card = Card()
        self.add_image_field(card)
        self.model = card.row('Model', ModelBox())
        self.start, self.end = spin(), spin()
        self.of = text('', 'hint')
        card.row('Timepoints', inline(self.start, text('to', 'sub'), self.end, self.of))
        self.out = card.row('Save results in', PathField(ctx.theme, 'folder', 'Next to the image'))
        self.seg_tif = QCheckBox('Also save the mask as <image>_SEG.tif')
        self.seg_tif.setChecked(True)
        card.row('', self.seg_tif)
        self.estimate = text('', 'hint')
        card.row('', self.estimate)
        self.body.addWidget(card)
        for s in (self.start, self.end):
            s.valueChanged.connect(self.update_estimate)
        self.add_run_bar('Run segmentation', self.submit)

    def image_changed(self, path):
        t = self.info['t'] if self.info else 1
        for s in (self.start, self.end):
            s.setRange(1, t)
        self.start.setValue(1)
        self.end.setValue(t)
        self.of.setText(f'of {t}' if self.info else '')
        self.update_estimate()

    def update_estimate(self):
        n = self.end.value() - self.start.value() + 1
        self.estimate.setText(f'Needs {disk_estimate(self.info, n, False)} of disk space for {n} timepoint'
                              f"{'s' if n != 1 else ''}." if self.info and n > 0 else '')

    def submit(self):
        if not self.info:
            return 'Choose an image first.'
        if not self.model.currentData():
            return 'No model is installed.'
        if self.end.value() < self.start.value():
            return 'The last timepoint must not be before the first.'
        path = self.image.path
        args = ['segment', path, '--model', self.model.currentData(), '--start', str(self.start.value()), '--end', str(self.end.value())]
        if self.out.path:
            args += ['--out', self.out.path]
        if not self.seg_tif.isChecked():
            args.append('--no-seg-tif')
        self.ctx.submit(Job(f'Segmentation  ·  {os.path.basename(path)}', args, ['segmentation'], image=path))


class TrainPage(Page):
    MODES = [('A new model, from scratch', 'new'), ('A new model, starting from an existing one', 'tune'),
             ('More epochs for an existing model', 'continue')]

    def __init__(self, ctx):
        super().__init__(ctx, 'Training', 'Teach a segmentation model your own images. Each training stack is a two-channel '
                                          'z stack: the image, and a mask of the objects drawn on it.')
        self.info = None
        card = Card()
        self.folder = card.row('Training stacks', PathField(ctx.theme, 'folder', 'Drop the folder of training stacks here, or browse'),
                               'The last 15 % of the stacks, in name order, are kept aside to score the model.')
        self.folder.changed.connect(self.folder_changed)
        self.mode = QComboBox()
        for label, key in self.MODES:
            self.mode.addItem(label, key)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        card.row('Train', self.mode)
        self.base = ModelBox()
        self.base_row = card.row('Existing model', self.base, 'Starting from a model trained on similar images needs far fewer epochs.')
        self.name = QLineEdit()
        card.row('Name of the model', self.name)
        self.epochs = spin(1, 100000, 100)
        card.row('Epochs', inline(self.epochs, text('passes through the training tiles', 'sub')))
        self.note = text('', 'hint', wrap=True)
        card.row('', self.note)
        self.body.addWidget(card)
        self.add_run_bar('Start training', self.submit)
        self.mode_changed()

    def folder_changed(self, path):
        self.info = None
        if path:
            try:
                from ..training import describe_folder
                self.info = describe_folder(path)
            except Exception as e:
                self.folder.set_detail(str(e))
                return
            i = self.info
            n_train, n_val = len(i['training_files']), len(i['validation_files'])
            self.folder.set_detail(i['problem'] or f"{n_train + n_val} stacks: {n_train} to train on, {n_val} to score  ·  "
                                                   f"{i['tiles_train']:,} training tiles  ·  {i['z']} z slices")
            if not self.name.text().strip():
                import datetime
                self.name.setText(f'{os.path.basename(path.rstrip("/"))}_{datetime.date.today():%Y%m%d}')
        self.update_note()

    def mode_changed(self, *_):
        mode = self.mode.currentData()
        self.base.setEnabled(mode != 'new')
        self.name.setEnabled(mode != 'continue')
        self.update_note()

    def update_note(self):
        parts = []
        if self.info and not self.info['problem']:
            parts.append(f"{self.info['tiles_train'] // 8:,} steps per epoch.")
        device = self.ctx.device_kind
        if device and device != 'cuda':
            parts.append('Training is slow without an NVIDIA GPU: expect seconds per step, so hours for a full training.')
        self.note.setText(' '.join(parts))

    def submit(self):
        mode = self.mode.currentData()
        if not self.info:
            return 'Choose the folder of training stacks.'
        if self.info['problem']:
            return self.info['problem']
        if mode != 'new' and not self.base.currentData():
            return 'No existing model is installed.'
        existing = {m['name'] for m in list_models(models_dir())}
        if mode == 'continue':
            name = os.path.splitext(os.path.basename(self.base.currentData()))[0]
        else:
            name = self.name.text().strip()
            if not name or any(ch in name for ch in '/\\:*?"<>|'):
                return 'Give the model a name (without / \\ : * ? " < > |).'
            if name in existing:
                return f'A model called {name} already exists. Choose another name.'
        args = ['train', self.folder.path, '--name', name, '--epochs', str(self.epochs.value())]
        if mode == 'tune':
            args += ['--from', self.base.currentData()]
        elif mode == 'continue':
            args.append('--continue')
        self.ctx.submit(Job(f'Training  ·  {name}', args, ['training'], folder=models_dir()))


class TrackPage(ImagePage):
    def __init__(self, ctx):
        super().__init__(ctx, 'Tracking', 'Follow each segmented object through time, with its splits and merges.')
        card = Card()
        self.add_image_field(card)
        self.seg = card.row('Segmentation results', PathField(ctx.theme, 'folder', 'The FC-DenseNet folder inside …_SegmentationOutput'),
                            'Filled in automatically when the results are next to the image.')
        self.seg.changed.connect(self.seg_changed)
        self.start, self.frames = spin(), spin(2, 99999, 2)
        self.of = text('', 'hint')
        card.row('Start at timepoint', inline(self.start, text('and track', 'sub'), self.frames, text('timepoints', 'sub'), self.of),
                 'Counted within the segmented range: 1 is the first segmented timepoint.')
        self.look_back = spin(1, 10, 2)
        card.row('Look back', inline(self.look_back, text('timepoints', 'sub')),
                 'How far back to search for an object that reappears. More than 2 is much slower.')
        self.keep = QCheckBox("Keep tracking's working files")
        card.row('', self.keep, 'The feature maps and the per-timepoint folders are only needed while tracking runs, and they are '
                                'most of the disk space. They are deleted when tracking has finished unless you tick this. '
                                'Without them, tracking this image again means running segmentation again first.')
        self.body.addWidget(card)
        self.add_run_bar('Run tracking', self.submit)

    def image_changed(self, path):
        folder = segmentation_folder(path) if path else ''
        self.seg.set_path(folder if os.path.isdir(folder) else '')

    def seg_changed(self, folder):
        n = segmented_frames(folder) if folder else None
        if n:
            self.start.setRange(1, max(1, n - 1))
            self.frames.setRange(2, n)
            self.frames.setValue(n)
        self.of.setText(f'of {n} segmented' if n else '')

    def submit(self):
        if not self.image.path:
            return 'Choose the image that was segmented.'
        if not self.seg.path or not os.path.isfile(os.path.join(self.seg.path, 'segmentation_parameters.csv')):
            return 'Choose the segmentation results folder (it contains segmentation_parameters.csv).'
        args = ['track', self.image.path, '--seg-folder', self.seg.path, '--start', str(self.start.value()),
                '--frames', str(self.frames.value()), '--trackback', str(self.look_back.value())]
        if self.keep.isChecked():
            args.append('--keep-working-folders')
        self.ctx.submit(Job(f'Tracking  ·  {os.path.basename(self.image.path)}', args, ['tracking'], image=self.image.path))


class AnalysisPage(ImagePage):
    def __init__(self, ctx):
        super().__init__(ctx, 'Analysis', 'Plots and tables of sizes, intensities and events from a tracked image.')
        card = Card()
        self.add_image_field(card)
        self.seg = card.row('Segmentation results', PathField(ctx.theme, 'folder', 'The FC-DenseNet folder inside …_SegmentationOutput'))
        self.track = card.row('Tracking results', PathField(ctx.theme, 'folder', 'The …_TrackingOutput folder'))
        self.seg.changed.connect(self.seg_changed)
        self.start, self.end = spin(), spin()
        card.row('Timepoints', inline(self.start, text('to', 'sub'), self.end),
                 'Counted within the tracked range.')
        self.body.addWidget(card)
        self.add_run_bar('Run analysis', self.submit)

    def image_changed(self, path):
        seg = segmentation_folder(path) if path else ''
        self.seg.set_path(seg if os.path.isdir(seg) else '')
        tracking = stages.tracking_folder(path, seg) if path else ''
        self.track.set_path(tracking if os.path.isdir(tracking) else '')

    def seg_changed(self, folder):
        n = segmented_frames(folder) if folder else None
        if n:
            for s in (self.start, self.end):
                s.setRange(1, n)
            self.start.setValue(1)
            self.end.setValue(n)

    def submit(self):
        if not self.image.path:
            return 'Choose the original image.'
        if not self.seg.path:
            return 'Choose the segmentation results folder.'
        if not self.track.path or not os.path.isfile(os.path.join(self.track.path, 'TrackedCombined.nii')):
            return 'Choose the tracking results folder (it contains TrackedCombined.nii).'
        args = ['analyze', self.image.path, '--seg-folder', self.seg.path, '--tracking-folder', self.track.path,
                '--start', str(self.start.value()), '--end', str(self.end.value())]
        self.ctx.submit(Job(f'Analysis  ·  {os.path.basename(self.image.path)}', args, ['analysis'],
                            image=self.image.path, folder=self.track.path))


class TreePage(Page):
    def __init__(self, ctx):
        super().__init__(ctx, 'Family tree', 'Draw the lineage of tracked objects: where each one came from and what it became.')
        card = Card()
        self.track = card.row('Tracking results', PathField(ctx.theme, 'folder', 'Drop the …_TrackingOutput folder here, or browse'))
        self.ids = QLineEdit()
        self.ids.setPlaceholderText('all')
        card.row('Object IDs', self.ids, 'One ID (76), a list (1,2,13), ranges (1-29, or 1-4,23-27), or all.')
        self.kind = QComboBox()
        self.kind.addItem('Future of each object', False)
        self.kind.addItem('Complete family', True)
        card.row('Tree', self.kind, 'Future: the object and what it becomes. Complete family: also its ancestors, everything '
                                    'that merged with it or with a relative, and all of their descendants.')
        self.overview = QCheckBox('Also draw one overview of every object in the movie')
        card.row('', self.overview)
        self.out = card.row('Save trees in', PathField(ctx.theme, 'folder', 'A FamilyTrees folder inside the tracking results'))
        self.body.addWidget(card)
        self.add_run_bar('Draw family trees', self.submit)

    def submit(self):
        if not self.track.path or not os.path.isfile(os.path.join(self.track.path, 'Sheet3.csv')):
            return 'Choose a tracking results folder (it contains Sheet3.csv).'
        args = ['tree', self.track.path, '--ids', self.ids.text().strip() or 'all']
        if self.kind.currentData():
            args.append('--complete')
        if self.overview.isChecked():
            args.append('--overview')
        if self.out.path:
            args += ['--out', self.out.path]
        name = os.path.basename(self.track.path.rstrip('/')).replace('_TrackingOutput', '')
        self.ctx.submit(Job(f'Family tree  ·  {name}', args, ['family tree'], folder=self.track.path))


class PipelinePage(Page):
    """Several images through segmentation, tracking and analysis, one after another."""
    COLUMNS = ['Image', 'Size (x × y × z)', 'Timepoints', 'First', 'Last']

    def __init__(self, ctx):
        super().__init__(ctx, 'Pipeline', 'Run segmentation, tracking and analysis on one image or a whole folder. '
                                          'Images are processed one after another.')
        self.rows = []  # (path, info)
        card = Card()
        top = QHBoxLayout()
        top.addWidget(text('Images', 'h2'))
        top.addStretch(1)
        for label, handler in (('Add images…', self.add_files), ('Add folder…', self.add_folder), ('Remove selected', self.remove_selected)):
            b = button(label)
            b.clicked.connect(handler)
            top.addWidget(b)
        card.box.addLayout(top)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        for col in (3, 4):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Fixed)
            self.table.setColumnWidth(col, 96)
        self.table.setMinimumHeight(190)
        self.table.verticalHeader().setDefaultSectionSize(40)
        card.add(self.table)
        self.empty = text('Drop images or a folder anywhere on this page, or use the buttons above. '
                          'A batch_times.txt in the folder sets the time ranges.', 'hint', wrap=True)
        card.add(self.empty)
        self.body.addWidget(card)
        self.setAcceptDrops(True)

        options = Card()
        self.model = options.row('Model', ModelBox())
        self.steps = {}
        boxes = []
        for key, label in (('seg', 'Segmentation'), ('track', 'Tracking'), ('analysis', 'Analysis')):
            box = QCheckBox(label)
            box.setChecked(True)
            self.steps[key] = box
            box.toggled.connect(self.update_estimate)
            boxes.append(box)
        self.estimate = text('', 'hint')
        options.row('Steps', inline(*boxes))
        self.trackback = spin(1, 10, 2)
        options.row('Look back', inline(self.trackback, text('timepoints when tracking', 'sub')))
        self.keep = QCheckBox("Keep tracking's working files (feature maps and per-timepoint folders)")
        self.keep.toggled.connect(self.update_estimate)
        options.row('', self.keep)
        options.row('', self.estimate)
        self.body.addWidget(options)
        self.add_run_bar('Run pipeline', self.submit)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.add_folder(path)
            else:
                self.add_image(path)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Choose images', '', IMAGE_FILTERS)
        for path in paths:
            self.add_image(path)

    def add_folder(self, folder=None):
        folder = folder or QFileDialog.getExistingDirectory(self, 'Choose a folder of images')
        if not folder:
            return
        times = batch.read_times(folder)
        for path in batch.list_images(folder):
            self.add_image(path, times)

    def add_image(self, path, times=None):
        if any(path == p for p, _ in self.rows):
            return
        try:
            info = read_image_info(path)
        except Exception as e:
            self.error.setText(str(e))
            return
        first, last = batch.time_range(path, times) if times else (1, info['t'])
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.rows.append((path, info))
        name = QTableWidgetItem(os.path.basename(path))
        name.setToolTip(path)
        self.table.setItem(row, 0, name)
        self.table.setItem(row, 1, QTableWidgetItem(f"{info['x']} × {info['y']} × {info['z']}"))
        self.table.setItem(row, 2, QTableWidgetItem(str(info['t'])))
        for col, value in ((3, first), (4, last)):
            box = spin(1, info['t'], max(1, min(info['t'], value)), 72)
            box.valueChanged.connect(self.update_estimate)
            holder = QWidget()
            holder.field = box
            centred = QHBoxLayout(holder)
            centred.setContentsMargins(0, 3, 0, 3)
            centred.addWidget(box, 0, Qt.AlignCenter)
            self.table.setCellWidget(row, col, holder)
        self.empty.hide()
        self.update_estimate()

    def remove_selected(self):
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
            del self.rows[row]
        self.empty.setVisible(not self.rows)
        self.update_estimate()

    def ranges(self):
        return [(self.table.cellWidget(r, 3).field.value(), self.table.cellWidget(r, 4).field.value()) for r in range(len(self.rows))]

    def update_estimate(self, *_):
        tracking = self.steps['track'].isChecked()
        peak, stays, largest = 0.0, 0.0, 0.0
        for (path, info), (first, last) in zip(self.rows, self.ranges()):
            n = max(0, last - first + 1)
            voxels = info['x'] * info['y'] * info['z']
            kept = voxels * 20 * n / 1e9                      # masks, labels and the tracked movie
            features = voxels * 64 * 4 * n / 1e9              # written by segmentation for tracking
            working = features + (voxels * 64 * 8 * n / 1e9 if tracking else 0.0)  # plus tracking's own folders
            stays += kept + (working if self.keep.isChecked() or not tracking else 0.0)
            largest = max(largest, working)
            peak += kept + working
        if not self.rows:
            self.estimate.setText('')
        elif tracking and not self.keep.isChecked():
            # images run one after another and each one's working folders are deleted before the next starts
            self.estimate.setText(f'Needs {size_text(stays + largest)} of free disk space while it runs; {size_text(stays)} stays afterwards.')
        else:
            self.estimate.setText(f'Needs {size_text(peak)} of disk space in total.')

    def submit(self):
        steps = [k for k, box in self.steps.items() if box.isChecked()]
        if not self.rows:
            return 'Add at least one image.'
        if not self.model.currentData():
            return 'No model is installed.'
        if not steps:
            return 'Choose at least one step.'
        for (path, info), (first, last) in zip(self.rows, self.ranges()):
            if last < first:
                return f'{os.path.basename(path)}: the last timepoint is before the first.'
            if 'track' in steps and last - first < 1:
                return f'{os.path.basename(path)}: tracking needs at least 2 timepoints.'
        for (path, info), (first, last) in zip(self.rows, self.ranges()):
            args = ['pipeline', path, '--model', self.model.currentData(), '--start', str(first), '--end', str(last),
                    '--trackback', str(self.trackback.value()), '--steps', ','.join(steps)]
            if self.keep.isChecked():
                args.append('--keep-working-folders')
            self.ctx.submit(Job(f'Pipeline  ·  {os.path.basename(path)}', args, [STEP_STAGE[s] for s in steps], image=path),
                            announce=False)
        self.ctx.toast(f'{len(self.rows)} image{"s" if len(self.rows) != 1 else ""} added to the queue')
        self.ctx.navigate('jobs')


class JobCard(Card):
    def __init__(self, ctx, job):
        super().__init__(margins=16, spacing=10)
        self.ctx, self.job = ctx, job
        top = QHBoxLayout()
        self.title = text(job.title, 'h2')
        self.state = text('', 'state_queued')
        top.addWidget(self.title, 1)
        top.addWidget(self.state)
        self.box.addLayout(top)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(8)
        self.chip = {}
        for stage in job.stages:
            self.chip[stage] = text(STAGE_LABELS.get(stage, stage), 'chip')
            self.chips.addWidget(self.chip[stage])
        self.chips.addStretch(1)
        self.box.addLayout(self.chips)
        self.bar = SmoothBar()
        self.box.addWidget(self.bar)
        bottom = QHBoxLayout()
        self.message = text('', 'sub')
        bottom.addWidget(self.message, 1)
        self.view_button = button('View', 'ghost')
        self.lineage_button = button('Lineage', 'ghost')
        self.lineage_button.clicked.connect(lambda: ctx.show_lineage(job.results['tracking']['tracking_folder']))
        self.folder_button = button('Open folder', 'ghost')
        self.log_button = button('Log', 'ghost')
        self.cancel_button = button('Cancel', 'danger')
        self.view_button.clicked.connect(lambda: ctx.view(job.image))
        self.folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(job.folder)))
        self.log_button.clicked.connect(self.toggle_log)
        self.cancel_button.clicked.connect(self.cancel_or_remove)
        for b in (self.view_button, self.lineage_button, self.folder_button, self.log_button, self.cancel_button):
            bottom.addWidget(b)
        self.box.addLayout(bottom)
        self.log = QPlainTextEdit()
        self.log.setObjectName('log')
        self.log.setReadOnly(True)
        self.log.setFixedHeight(190)
        self.log.hide()
        self.box.addWidget(self.log)
        job.changed.connect(self.refresh)
        self.refresh()

    def toggle_log(self):
        self.log.setVisible(not self.log.isVisible())
        self.refresh()

    def cancel_or_remove(self):
        if self.job.state in ('queued', 'running'):
            self.ctx.jobs.cancel(self.job)
        else:
            self.ctx.jobs.remove(self.job)

    def refresh(self):
        job, c = self.job, self.ctx.theme.colors
        rename(self.state, f'state_{job.state}')
        self.state.setText({'queued': 'Queued', 'running': 'Running', 'done': 'Done', 'failed': 'Failed', 'cancelled': 'Cancelled'}[job.state])
        marks = {'pending': '', 'running': '  …', 'done': '  ✓', 'failed': '  ✕', 'cancelled': '  –'}
        for stage, label in self.chip.items():
            label.setText(STAGE_LABELS.get(stage, stage) + marks[job.stage_state[stage]])
        if job.state == 'running':
            self.bar.set_fraction(job.done / job.total if job.total else None)
        else:
            self.bar.set_fraction(1.0 if job.state == 'done' else 0.0 if job.state == 'queued' else self.bar.value() / 1000)
        parts = [job.message] if job.message else []
        if job.epochs:
            last = job.epochs[-1]
            parts.append(f"validation IoU {last['validation_iou']:.3f} (best {last['best_iou']:.3f} at epoch {last['best_epoch']})")
        if job.state == 'running':
            if job.total:
                parts.insert(0, f'{STAGE_LABELS.get(job.stage, job.stage)} {job.done} of {job.total}')
            if job.eta and job.eta > 0:
                parts.append(f'about {format_seconds(job.eta)} left in this step')
        if job.started:
            parts.append(f'{format_seconds(job.elapsed())} elapsed' if job.state == 'running' else f'took {format_seconds(job.elapsed())}')
        self.message.setText('   ·   '.join(parts))
        self.message.setToolTip(job.error or '')
        self.view_button.setVisible(bool(job.image) and job.state == 'done')
        self.lineage_button.setVisible('tracking' in job.results)
        self.folder_button.setVisible(bool(job.folder) and os.path.isdir(job.folder or ''))
        self.cancel_button.setText('Cancel' if job.state in ('queued', 'running') else 'Remove')
        if self.log.isVisible():
            self.log.setPlainText('\n'.join(job.log[-400:]))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


class JobsPage(Page):
    def __init__(self, ctx):
        super().__init__(ctx, 'Jobs', 'Everything that is running, waiting or finished in this session.')
        self.cards = {}
        self.empty = text('Nothing has been run yet. Start from Pipeline, or from one of the single steps.', 'hint')
        self.body.addWidget(self.empty)
        self.list = QVBoxLayout()
        self.list.setSpacing(12)
        self.body.addLayout(self.list)
        self.body.addStretch(1)
        ctx.jobs.added.connect(self.add)
        ctx.jobs.changed.connect(self.sync)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    def add(self, job):
        card = JobCard(self.ctx, job)
        self.cards[job] = card
        self.list.insertWidget(0, card)
        self.empty.hide()

    def sync(self):
        for job in [j for j in self.cards if j not in self.ctx.jobs.jobs]:
            self.cards.pop(job).deleteLater()
        self.empty.setVisible(not self.cards)

    def tick(self):
        for job, card in self.cards.items():
            if job.state == 'running':
                card.refresh()


class ModelsPage(Page):
    def __init__(self, ctx):
        super().__init__(ctx, 'Models', 'Trained segmentation models. Each one was trained on one kind of image; '
                                        'pick the one that matches your protein.')
        self.grid = QGridLayout()
        self.grid.setSpacing(14)
        self.body.addLayout(self.grid)
        self.empty = text('No models are installed yet. Download the published ones, add model files (.pt), or train '
                          'one on the Training page.', 'hint', wrap=True)
        self.body.addWidget(self.empty)
        row = QHBoxLayout()
        self.download_button = button('Download published models', 'primary')
        self.download_button.clicked.connect(self.download)
        row.addWidget(self.download_button)
        add_button = button('Add model files…')
        add_button.clicked.connect(self.add_models)
        open_button = button('Open models folder')
        open_button.clicked.connect(self.open_folder)
        row.addWidget(add_button)
        row.addWidget(open_button)
        row.addStretch(1)
        self.body.addLayout(row)
        self.body.addStretch(1)
        self.reload()

    def download(self):
        self.ctx.submit(Job('Download  ·  published models', ['download-models'] + self.missing, ['download'], folder=models_dir()))

    def open_folder(self):
        os.makedirs(models_dir(), exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(models_dir()))

    def add_models(self):
        """Copy chosen model files, with their description files, into the models folder."""
        import shutil
        paths, _ = QFileDialog.getOpenFileNames(self, 'Choose model files', '', 'ProTrack3D models (*.pt)')
        os.makedirs(models_dir(), exist_ok=True)
        added = 0
        for path in paths:
            if os.path.dirname(os.path.abspath(path)) == os.path.abspath(models_dir()):
                continue
            shutil.copy2(path, models_dir())
            if os.path.isfile(os.path.splitext(path)[0] + '.json'):
                shutil.copy2(os.path.splitext(path)[0] + '.json', models_dir())
            added += 1
        if added:
            self.ctx.models_changed()
            self.ctx.toast(f'{added} model{"s" if added != 1 else ""} added')

    def reload(self):
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        models = list_models(models_dir())
        self.empty.setVisible(not models)
        from ..catalog import load_catalog
        published = [m for m in load_catalog()['models'] if not m['installed']]
        self.missing = [m['name'] for m in published]
        self.download_button.setVisible(bool(published))
        self.download_button.setText(f"Download {len(published)} published model{'s' if len(published) != 1 else ''}  "
                                     f"({sum(m['size'] for m in published) / 1e6:.0f} MB)")
        for i, m in enumerate(models):
            card = Card(margins=16, spacing=6)
            card.add(text(m['name'], 'h2'))
            facts = []
            if m.get('best_val_iou'):
                facts.append(f"Validation IoU {m['best_val_iou']:.2f}" + (f" at epoch {m['best_epoch']}" if m.get('best_epoch') is not None else ''))
            if m.get('trained_on'):
                facts.append(f"Trained {m['trained_on']}")
            facts.append(f"{m.get('architecture', '')}  ·  {m['size_mb']:.0f} MB")
            for fact in facts:
                card.add(text(fact, 'sub'))
            self.grid.addWidget(card, i // 2, i % 2)


class HomePage(Page):
    ACTIONS = [('pipeline', 'Pipeline', 'Segmentation, tracking and analysis for one image or a folder.'),
               ('segment', 'Segmentation', 'Find the objects in each timepoint.'),
               ('track', 'Tracking', 'Follow objects through time.'),
               ('tree', 'Family tree', 'Draw the lineage of chosen objects.'),
               ('analysis', 'Analysis', 'Plots and tables from tracked objects.'),
               ('viewer', 'Viewer', 'Look through an image with its masks and track IDs.')]

    def __init__(self, ctx):
        super().__init__(ctx, 'ProTrack3D', 'Segment, track and analyse protein clusters in 3D time-lapse microscopy.')
        grid = QGridLayout()
        grid.setSpacing(14)
        self.buttons = []
        for i, (key, title, blurb) in enumerate(self.ACTIONS):
            b = QPushButton()
            b.setObjectName('action')
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(136)
            inner = QVBoxLayout(b)
            inner.setContentsMargins(18, 16, 18, 16)
            inner.setSpacing(6)
            pic = QLabel()
            pic.setAttribute(Qt.WA_TransparentForMouseEvents)
            inner.addWidget(pic)
            for label in (text(title, 'h2'), text(blurb, 'sub', wrap=True)):
                label.setAttribute(Qt.WA_TransparentForMouseEvents)
                inner.addWidget(label)
            inner.addStretch(1)
            b.clicked.connect(lambda _=False, k=key: ctx.navigate(k))
            self.buttons.append((pic, key))
            grid.addWidget(b, i // 3, i % 3)
        self.body.addLayout(grid)
        self.recent = Card('Recent jobs')
        self.recent_text = self.recent.add(text('Nothing has been run in this session.', 'hint', wrap=True))
        self.body.addWidget(self.recent)
        self.updates = Card('Updates')
        self.update_text = self.updates.add(text('', 'sub', wrap=True))
        self.update_news = self.updates.add(text('', 'hint', wrap=True))
        row = QHBoxLayout()
        self.update_action = button('', 'primary')
        self.update_action.clicked.connect(self.update_clicked)
        self.update_check = button('Check for updates')
        self.update_check.clicked.connect(ctx.updater.check)
        row.addWidget(self.update_action)
        row.addWidget(self.update_check)
        row.addStretch(1)
        self.updates.box.addLayout(row)
        self.body.addWidget(self.updates)
        self.body.addStretch(1)
        ctx.updater.changed.connect(self.show_update)
        self.show_update()
        ctx.theme.listeners.append(self.restyle)
        ctx.jobs.changed.connect(self.refresh)
        ctx.jobs.finished.connect(lambda _job: self.refresh())
        self.restyle()

    def restyle(self):
        for pic, key in self.buttons:
            pic.setPixmap(icons.pixmap(key, self.ctx.theme.colors['accent'], 24))

    def show_update(self):
        u = self.ctx.updater
        info, state = u.info, u.state

        def version(v):
            return f"{v['id']} from {v['date']}" if v else 'version unknown'

        self.update_text.setText({
            'unknown': 'Not checked yet.',
            'checking': 'Looking for a newer version…',
            'current': f"You have the newest version ({version(info.get('current'))}).",
            'available': f"A newer version is available ({version(info.get('latest'))}).",
            'updating': 'Updating. The Jobs page shows how far it is.',
            'updated': 'Updated. Start ProTrack3D again to use the new version.',
            'error': f'That did not work: {u.error}',
            'packaged': 'This is the packaged app. It cannot update itself; get the new version from GitHub.'}[state])
        rename(self.update_text, 'error' if state == 'error' else 'sub')
        news = ''
        if state == 'available':
            changes = info.get('changes', [])
            if info.get('kind') == 'zip' and info.get('files'):
                news = f"{info['files']} file{'s' if info['files'] != 1 else ''} will be replaced. Newest change: {changes[0]}\n"
            else:
                news = ''.join(f'•  {c}\n' for c in changes[:8]) + (f'and {len(changes) - 8} more\n' if len(changes) > 8 else '')
            news += 'Your models, images and results are not touched.'
        self.update_news.setText(news)
        self.update_news.setVisible(bool(news))
        self.update_action.setText('Restart ProTrack3D' if state == 'updated' else 'Update now')
        self.update_action.setVisible(state in ('available', 'updated'))
        self.update_check.setVisible(state in ('unknown', 'checking', 'current', 'available', 'error'))
        self.update_check.setEnabled(state != 'checking')

    def update_clicked(self):
        u = self.ctx.updater
        problem = u.restart() if u.state == 'updated' else u.install()
        if problem:
            self.ctx.toast(problem)

    def refresh(self):
        jobs = self.ctx.jobs.jobs[-5:][::-1]
        labels = {'queued': 'queued', 'running': 'running', 'done': 'done', 'failed': 'failed', 'cancelled': 'cancelled'}
        self.recent_text.setText('\n'.join(f'{j.title}   —   {labels[j.state]}' for j in jobs) or 'Nothing has been run in this session.')
