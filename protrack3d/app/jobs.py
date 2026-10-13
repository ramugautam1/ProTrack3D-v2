"""Jobs run the engine as a separate process, one at a time, and turn its progress events into Qt signals."""
import json
import os
import sys
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ..events import MARKER
from ..paths import FROZEN, REPO_ROOT

STAGE_LABELS = {'segmentation': 'Segmentation', 'tracking': 'Tracking', 'analysis': 'Analysis', 'family tree': 'Family tree',
                'training': 'Training', 'download': 'Download', 'update': 'Update', 'preparation': 'Preparation',
                'quantification': 'Measuring', 'boundary analysis': 'Boundary analysis'}


def engine_command():
    """How to start the engine. A packaged app re-launches itself; from source it is `python -m protrack3d`."""
    if getattr(sys, 'frozen', False):
        return sys.executable, ['--engine']
    python = sys.executable
    if os.path.basename(python).lower() == 'pythonw.exe':  # the windowless Python has no output to read
        python = os.path.join(os.path.dirname(python), 'python.exe')
    return python, ['-W', 'ignore', '-m', 'protrack3d']


class Job(QObject):
    changed = Signal()

    def __init__(self, title, args, stages, image=None, folder=None):
        super().__init__()
        self.title, self.args, self.stages = title, list(args), list(stages)
        self.image, self.folder = image, folder
        self.state = 'queued'  # queued, running, done, failed, cancelled
        self.stage_state = {s: 'pending' for s in stages}
        self.stage, self.done, self.total, self.message, self.eta = None, 0, None, 'Waiting', None
        self.results, self.log, self.error = {}, [], None
        self.epochs = []  # one record per finished training epoch
        self.started = self.finished = None

    def elapsed(self):
        if not self.started:
            return 0
        return (self.finished or time.time()) - self.started

    def handle(self, event):
        kind = event.get('event')
        if kind == 'stage':
            self.stage = event['stage']
            self.stage_state[self.stage] = 'running'
            self.done, self.total, self.eta, self.message = 0, event.get('total'), None, event.get('message', '')
        elif kind == 'progress':
            self.done, self.total = event['done'], event['total'] or None
            self.message, self.eta = event.get('message', ''), event.get('eta')
        elif kind == 'stage_done':
            self.stage_state[event['stage']] = 'done'
            self.results[event['stage']] = event
            for key in ('analysis_folder', 'family_tree_folder', 'tracking_folder', 'segmentation_folder'):
                if key in event:
                    self.folder = event[key]
                    break
            if event.get('image'):
                self.image = event['image']
        elif kind == 'epoch':
            self.epochs.append(event)
        elif kind == 'error':
            self.error = event.get('message')
            if event.get('stage') in self.stage_state:
                self.stage_state[event['stage']] = 'failed'
        self.changed.emit()


class JobManager(QObject):
    added = Signal(object)
    started = Signal(object)
    output = Signal(object, str)   # a line the engine printed
    event = Signal(object, dict)   # a progress event from the engine
    finished = Signal(object)
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.jobs, self.current, self.process, self._buffer = [], None, None, b''

    def submit(self, job):
        self.jobs.append(job)
        self.added.emit(job)
        self.changed.emit()
        self._next()
        return job

    def active_count(self):
        return sum(j.state in ('queued', 'running') for j in self.jobs)

    def cancel(self, job):
        if job.state == 'queued':
            job.state, job.message = 'cancelled', 'Cancelled'
            job.changed.emit()
            self.changed.emit()
        elif job is self.current and self.process:
            job.state = 'cancelled'
            self.process.kill()

    def remove(self, job):
        if job.state not in ('queued', 'running') and job in self.jobs:
            self.jobs.remove(job)
            self.changed.emit()

    def _next(self):
        if self.current:
            return
        job = next((j for j in self.jobs if j.state == 'queued'), None)
        if not job:
            return
        self.current, self._buffer = job, b''
        job.state, job.started, job.message = 'running', time.time(), 'Starting'
        program, prefix = engine_command()
        self.process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(os.path.expanduser('~') if FROZEN else REPO_ROOT)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._failed_to_start)
        self.process.start(program, prefix + ['--json'] + job.args)
        self.started.emit(job)
        job.changed.emit()
        self.changed.emit()

    def _read(self):
        self._buffer += bytes(self.process.readAllStandardOutput())
        *lines, self._buffer = self._buffer.replace(b'\r', b'\n').split(b'\n')
        job = self.current
        for raw in lines:
            line = raw.decode('utf-8', 'replace').rstrip()
            at = line.find(MARKER)
            if at >= 0:
                try:
                    event = json.loads(line[at + len(MARKER):])
                    job.handle(event)
                    self.event.emit(job, event)
                except ValueError:
                    pass
                line = line[:at]
            if line.strip():
                job.log.append(line)
                del job.log[:-3000]
                self.output.emit(job, line)

    def _failed_to_start(self, error):
        if error == QProcess.FailedToStart and self.current:
            self.current.error = 'The engine could not be started.'
            self._finished(1, QProcess.CrashExit)

    def _finished(self, code, _status):
        job = self.current
        if not job:
            return
        if self.process:
            self._read()
        job.finished = time.time()
        if job.state != 'cancelled':
            job.state = 'done' if code == 0 else 'failed'
        if job.state == 'done':
            job.message = 'Finished'
        elif job.state == 'cancelled':
            job.message = 'Cancelled'
            for s, v in job.stage_state.items():
                if v == 'running':
                    job.stage_state[s] = 'cancelled'
        else:
            job.message = job.error or (job.log[-1] if job.log else 'Failed')
            for s, v in job.stage_state.items():
                if v == 'running':
                    job.stage_state[s] = 'failed'
        self.current, self.process = None, None
        job.changed.emit()
        self.finished.emit(job)
        self.changed.emit()
        self._next()


def format_seconds(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds} s'
    if seconds < 3600:
        return f'{seconds // 60} min {seconds % 60:02d} s'
    return f'{seconds // 3600} h {seconds % 3600 // 60:02d} min'
