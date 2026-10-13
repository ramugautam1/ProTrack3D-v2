import json
import os
import sys

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter, QVBoxLayout, QWidget

from ..paths import FROZEN, REPO_ROOT
from . import icons
from .armdia_page import ArmDiaPage
from .console import Console
from .jobs import JobManager, engine_command
from .lineage_page import LineagePage
from .updater import Updater
from .pages import AnalysisPage, HomePage, JobsPage, ModelsPage, PipelinePage, SegmentPage, TrackPage, TrainPage, TreePage
from .widgets import ModelBox
from .viewer import ViewerPage
from .widgets import FadeStack, Toast, text

NAV = [('home', 'Home', HomePage), ('pipeline', 'Pipeline', PipelinePage), ('segment', 'Segmentation', SegmentPage),
       ('track', 'Tracking', TrackPage), ('tree', 'Family tree', TreePage), ('lineage', 'Lineage', LineagePage),
       ('analysis', 'Analysis', AnalysisPage),
       ('armdia', 'Armadillo / Dia', ArmDiaPage), ('viewer', 'Viewer', ViewerPage), ('train', 'Training', TrainPage),
       ('models', 'Models', ModelsPage),
       ('jobs', 'Jobs', JobsPage)]


class MainWindow(QMainWindow):
    def __init__(self, theme):
        super().__init__()
        self.theme, self.jobs = theme, JobManager()
        self.device_kind = None  # 'cuda', 'mps' or 'cpu', once the engine has answered
        self.updater = Updater(self)
        self.setWindowTitle('ProTrack3D')
        self.resize(1180, 800)
        self.setMinimumSize(980, 640)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        side = QWidget()
        side.setObjectName('sidebar')
        side.setFixedWidth(214)
        bar = QVBoxLayout(side)
        bar.setContentsMargins(14, 22, 14, 14)
        bar.setSpacing(3)
        bar.addWidget(text('ProTrack3D', 'brand'))
        bar.addWidget(text('3D segmentation and tracking', 'brandSub'))
        bar.addSpacing(18)

        self.stack = FadeStack()
        self.stack.setObjectName('content')
        self.group = QButtonGroup(self)
        self.buttons, self.pages = {}, {}
        for i, (key, label, page_class) in enumerate(NAV):
            if key == 'jobs':
                bar.addStretch(1)
            b = QPushButton('  ' + label)
            b.setObjectName('nav')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.toggled.connect(self.restyle)
            b.clicked.connect(lambda _=False, k=key: self.navigate(k))
            self.group.addButton(b)
            self.buttons[key] = b
            if key == 'jobs':
                row = QHBoxLayout(b)
                row.setContentsMargins(0, 0, 10, 0)
                row.addStretch(1)
                self.badge = QLabel('')
                self.badge.setObjectName('badge')
                self.badge.setFixedHeight(18)
                self.badge.setAlignment(Qt.AlignCenter)
                self.badge.setAttribute(Qt.WA_TransparentForMouseEvents)
                self.badge.hide()
                row.addWidget(self.badge)
            bar.addWidget(b)
            self.pages[key] = page_class(self)
            self.stack.addWidget(self.pages[key])
        self.update_notice = QPushButton()
        self.update_notice.setObjectName('notice')
        self.update_notice.setCursor(Qt.PointingHandCursor)
        self.update_notice.clicked.connect(lambda: self.navigate('home'))
        self.update_notice.hide()
        bar.addSpacing(6)
        bar.addWidget(self.update_notice)
        bar.addSpacing(10)
        foot = QHBoxLayout()
        self.device = text('', 'chip')
        self.device.hide()
        foot.addWidget(self.device)
        foot.addStretch(1)
        self.console_button = QPushButton()
        self.console_button.setObjectName('ghost')
        self.console_button.setCheckable(True)
        self.console_button.setToolTip('Show or hide the console (Ctrl+`)')
        self.console_button.setShortcut('Ctrl+`')
        self.console_button.setCursor(Qt.PointingHandCursor)
        self.console_button.toggled.connect(self.show_console)
        foot.addWidget(self.console_button)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName('ghost')
        self.theme_button.setToolTip('Switch between light and dark')
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.clicked.connect(theme.toggle)
        foot.addWidget(self.theme_button)
        bar.addLayout(foot)

        self.console = Console(self)
        self.console.closed.connect(lambda: self.console_button.setChecked(False))
        self.split = QSplitter(Qt.Vertical)
        self.split.setObjectName('content')
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.stack)
        self.split.addWidget(self.console)
        self.split.setStretchFactor(0, 1)
        self.console.hide()
        layout.addWidget(side)
        layout.addWidget(self.split, 1)
        self.toaster = Toast(root)

        self.jobs.changed.connect(self.update_badge)
        self.jobs.finished.connect(self.job_finished)
        theme.listeners.append(self.restyle)
        self.buttons['home'].setChecked(True)
        self.restyle()
        self.ask_device()
        self.updater.changed.connect(self.update_changed)
        QTimer.singleShot(1500, lambda: self.updater.check(quiet=True))  # once the window is up

    # --- what pages use ---
    def navigate(self, key):
        self.buttons[key].setChecked(True)
        self.stack.show_page([k for k, _, _ in NAV].index(key))

    def toast(self, message):
        self.toaster.announce(message)

    def submit(self, job, announce=True):
        self.jobs.submit(job)
        if announce:
            self.toast('Added to the queue' if self.jobs.active_count() > 1 else 'Started')
            self.navigate('jobs')

    def view(self, image):
        self.navigate('viewer')
        self.pages['viewer'].show_image(image)

    def show_lineage(self, tracking_folder, obj=None):
        self.navigate('lineage')
        self.pages['lineage'].show_object(tracking_folder, obj)

    def view_object(self, lineage, obj):
        """Open an object from the lineage in the viewer. The image is looked for next to the tracking results."""
        viewer = self.pages['viewer']
        image = lineage.image_path()
        if image is None and viewer.stack and os.path.normpath(viewer.stack.tracking_folder) == os.path.normpath(lineage.folder):
            image = viewer.stack.path
        if image is None:
            from PySide6.QtWidgets import QFileDialog
            image, _ = QFileDialog.getOpenFileName(self, 'Choose the image this tracking belongs to', os.path.dirname(lineage.folder))
        if image:
            self.navigate('viewer')
            viewer.show_object(image, lineage, obj)

    def show_console(self, shown):
        if shown and not self.console.isVisible():
            height = int(QSettings().value('console_height', 220))
            self.console.show()
            self.split.setSizes([max(200, self.split.height() - height), height])
        elif not shown and self.console.isVisible():
            QSettings().setValue('console_height', self.console.height())
            self.console.hide()
        self.restyle()

    # --- window chrome ---
    def restyle(self, *_):
        c = self.theme.colors
        for key, b in self.buttons.items():
            b.setIcon(icons.icon(key, c['accent'] if b.isChecked() else c['muted'], 18))
        self.theme_button.setIcon(icons.icon('sun' if self.theme.dark else 'moon', c['muted'], 16))
        self.console_button.setIcon(icons.icon('terminal', c['accent'] if self.console_button.isChecked() else c['muted'], 16))

    def update_badge(self):
        n = self.jobs.active_count()
        self.badge.setText(str(n))
        self.badge.setVisible(n > 0)

    def job_finished(self, job):
        words = {'done': 'finished', 'failed': 'failed', 'cancelled': 'was cancelled'}
        self.toast(f'{job.title}  {words.get(job.state, job.state)}')
        if 'training' in job.stages or 'download' in job.stages:
            self.models_changed()

    def update_changed(self):
        state = self.updater.state
        self.update_notice.setText({'available': 'Update available', 'updated': 'Restart to finish updating'}.get(state, ''))
        self.update_notice.setVisible(state in ('available', 'updated'))
        if state == 'updated':
            self.navigate('home')

    def models_changed(self):
        """A model was added or updated: refresh every list of models."""
        for box in self.findChildren(ModelBox):
            box.reload()
        self.pages['models'].reload()

    def ask_device(self):
        program, prefix = engine_command()
        probe = QProcess(self)
        probe.setWorkingDirectory(os.path.expanduser('~') if FROZEN else REPO_ROOT)

        def done(*_):
            try:
                answer = json.loads(bytes(probe.readAllStandardOutput()).decode().strip().splitlines()[-1])
                label, self.device_kind = answer['label'], answer['device']
                self.pages['train'].update_note()
                self.device.setText(label)
                self.device.setToolTip(f'Segmentation runs on the {label}')
                self.device.show()
            except (ValueError, IndexError, KeyError, RuntimeError):  # RuntimeError: the window closed first
                pass

        probe.finished.connect(done)
        probe.start(program, prefix + ['device'])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.toaster.isVisible():
            self.toaster.move(self.centralWidget().width() - self.toaster.width() - 24,
                              self.centralWidget().height() - self.toaster.height() - 24)

    def closeEvent(self, event):
        if self.jobs.current:
            self.jobs.cancel(self.jobs.current)
        event.accept()
