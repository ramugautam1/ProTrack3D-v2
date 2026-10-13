"""Armadillo / Dia: the fixed-embryo workflow. Measure Dia inside the Armadillo puncta, then sort puncta by cell boundary."""
import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit

from ..armdia import describe_folder
from .jobs import Job
from .pages import Page, inline, spin
from .widgets import Card, ModelBox, PathField, button, text


class ArmDiaPage(Page):
    def __init__(self, ctx):
        super().__init__(ctx, 'Armadillo / Dia', 'For fixed embryos imaged once, with several channels in one file. Armadillo puncta '
                                                 'are found, and Dia is measured inside and outside them.')
        self.info = None
        card = Card('Measure')
        self.folder = card.row('Folder of images', PathField(ctx.theme, 'folder', 'Drop the folder here (.czi or .tif, subfolders included), or browse'),
                               'Results are written next to each image, and one summary table into this folder.')
        self.folder.changed.connect(self.folder_changed)
        self.model = card.row('Model', ModelBox())
        bazooka = self.model.findText('BAZOOKA', flags=Qt.MatchContains)  # the model this workflow has been run with
        if bazooka >= 0 and not QSettings().value('last_model', ''):
            self.model.setCurrentIndex(bazooka)
        self.arm_channel, self.dia_channel = spin(1, 99, 2, 56), spin(1, 99, 3, 56)
        self.arm_only = QCheckBox('The images hold Armadillo only')
        self.arm_only.toggled.connect(lambda on: self.dia_channel.setEnabled(not on))
        card.row('Channels', inline(text('Armadillo is channel', 'sub'), self.arm_channel, text('Dia is channel', 'sub'), self.dia_channel, self.arm_only))
        self.z_layers, self.size_filter = spin(1, 9999, 20, 56), spin(0, 999999, 160, 70)
        card.row('Slices', inline(text('Use the first', 'sub'), self.z_layers, text('z slices', 'sub')))
        card.row('Smallest object', inline(text('Remove puncta of', 'sub'), self.size_filter, text('voxels or fewer', 'sub')))
        self.threshold = QCheckBox('Before measuring, set everything at or below background (mean + 3 standard deviations) to zero')
        self.threshold.setChecked(True)
        card.row('Background', self.threshold, 'Background is everything outside the Armadillo puncta.')
        self.table = QLineEdit()
        self.table.setPlaceholderText('ARM_DIA_Summary.csv')
        card.row('Name of the table', self.table)
        self.body.addWidget(card)
        self.add_run_bar('Run', self.submit)
        self.body.takeAt(self.body.count() - 1)  # the stretch; the second card goes below the first

        second = Card('Puncta on the cell boundary')
        second.add(text('Needs a "Cell Outlines" folder next to the embryo folders, with one outline image per stack. Each outline is '
                        'thickened, and a punctum counts as a boundary punctum when enough of it lies on the outline. Run "Measure" first.', 'sub', wrap=True))
        self.outlines = second.row('Outlines folder', PathField(ctx.theme, 'folder', '"Cell Outlines" inside the folder above'))
        self.current, self.target, self.overlap = spin(1, 999, 6, 56), spin(1, 999, 30, 56), spin(1, 100, 30, 56)
        second.row('Outline width', inline(text('Drawn at', 'sub'), self.current, text('pixels, thicken to', 'sub'), self.target, text('pixels', 'sub')))
        second.row('On the boundary when', inline(self.overlap, text('% or more of the punctum overlaps', 'sub')))
        self.body.addWidget(second)
        bar = QHBoxLayout()
        self.boundary_error = text('', 'error', wrap=True)
        run = button('Run boundary analysis', 'primary')
        run.clicked.connect(lambda: self.boundary_error.setText(self.submit_boundary() or ''))
        bar.addWidget(self.boundary_error, 1)
        bar.addWidget(run)
        self.body.addLayout(bar)
        self.body.addStretch(1)

    def folder_changed(self, path):
        self.info = None
        if path:
            try:
                self.info = describe_folder(path)
            except Exception as e:
                self.folder.set_detail(str(e))
                return
            first = self.info['first']
            if first:
                self.folder.set_detail(f"{self.info['images']} images in {self.info['folders']} folders  ·  the first has {first['channels']} "
                                       f"channel{'s' if first['channels'] != 1 else ''}, {first['z']} slices, {first['x']} × {first['y']}")
                if first['channels'] == 1:
                    self.arm_only.setChecked(True)
                    self.arm_channel.setValue(1)
            else:
                self.folder.set_detail('No .czi or .tif images in this folder.')

    def submit(self):
        if not self.info or not self.info['images']:
            return 'Choose a folder that holds .czi or .tif images.'
        if not self.model.currentData():
            return 'No model is installed.'
        channels = self.info['first']['channels']
        wanted = [self.arm_channel.value()] + ([] if self.arm_only.isChecked() else [self.dia_channel.value()])
        if max(wanted) > channels:
            return f'The images have {channels} channel{"s" if channels != 1 else ""}; channel {max(wanted)} does not exist.'
        if not self.arm_only.isChecked() and self.arm_channel.value() == self.dia_channel.value():
            return 'Armadillo and Dia cannot be the same channel.'
        args = ['armdia', self.folder.path, '--model', self.model.currentData(), '--z-layers', str(self.z_layers.value()),
                '--size-filter', str(self.size_filter.value()), '--arm-channel', str(self.arm_channel.value()),
                '--dia-channel', str(self.dia_channel.value())]
        if self.arm_only.isChecked():
            args.append('--arm-only')
        if not self.threshold.isChecked():
            args.append('--no-threshold')
        name = self.table.text().strip()
        if name:
            args += ['--csv', name if name.lower().endswith('.csv') else name + '.csv']
        self.ctx.submit(Job(f'Armadillo / Dia  ·  {os.path.basename(self.folder.path.rstrip("/"))}', args,
                            ['preparation', 'segmentation', 'quantification'], folder=self.folder.path))

    def submit_boundary(self):
        if not self.folder.path:
            return 'Choose the folder of images at the top first.'
        outlines = self.outlines.path or os.path.join(self.folder.path, 'Cell Outlines')
        if not os.path.isdir(outlines):
            return f'There is no outlines folder at {outlines}.'
        if self.target.value() <= self.current.value():
            return 'The thickened width must be larger than the drawn width.'
        args = ['armdia-boundary', self.folder.path, '--outlines', outlines, '--current-width', str(self.current.value()),
                '--target-width', str(self.target.value()), '--overlap', str(self.overlap.value() / 100)]
        self.ctx.submit(Job(f'Boundary puncta  ·  {os.path.basename(self.folder.path.rstrip("/"))}', args, ['boundary analysis'], folder=self.folder.path))
