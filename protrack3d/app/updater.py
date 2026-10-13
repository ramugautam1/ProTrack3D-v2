"""Looks for a newer published version, installs it as a job, and starts the app again."""
import json
import sys

from PySide6.QtCore import QObject, QProcess, Signal
from PySide6.QtWidgets import QApplication

from ..events import MARKER
from ..paths import FROZEN, REPO_ROOT
from .jobs import Job, engine_command


class Updater(QObject):
    changed = Signal()

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx = ctx
        # unknown, checking, current, available, updating, updated, error, or packaged (the packaged app cannot update itself)
        self.state = 'packaged' if FROZEN else 'unknown'
        self.info, self.error, self.job, self.probe, self.quiet = {}, '', None, None, False
        ctx.jobs.finished.connect(self._job_finished)

    def _set(self, state, error=''):
        self.state, self.error = state, error
        self.changed.emit()

    def check(self, quiet=False):
        """Ask the engine whether a newer version is published. A quiet check (the one at start-up) says nothing when it fails."""
        if self.state in ('packaged', 'checking', 'updating', 'updated'):
            return
        self.quiet = quiet is True
        program, prefix = engine_command()
        self.probe = QProcess(self)
        self.probe.setWorkingDirectory(REPO_ROOT)
        self.probe.setProcessChannelMode(QProcess.MergedChannels)
        self.probe.finished.connect(self._checked)
        self.probe.errorOccurred.connect(lambda _e: self.state == 'checking' and self._failed('The engine could not be started.'))
        self._set('checking')
        self.probe.start(program, prefix + ['--json', 'update', '--check'])

    def _failed(self, message):
        self._set('unknown') if self.quiet else self._set('error', message)

    def _checked(self, *_):
        if self.state != 'checking':
            return
        answer, error = None, ''
        for line in bytes(self.probe.readAllStandardOutput()).decode('utf-8', 'replace').splitlines():
            try:
                if MARKER in line:
                    event = json.loads(line[line.find(MARKER) + len(MARKER):])
                    if event.get('event') == 'error':
                        error = event.get('message', '')
                elif line.startswith('{'):
                    answer = json.loads(line)
            except ValueError:
                pass
        if answer is None:
            return self._failed(error.split(': ', 1)[-1] or 'Could not look for a newer version.')
        self.info = answer
        self._set('available' if answer.get('available') else 'current')

    def install(self):
        """Start the update. Returns a message when it cannot be started."""
        if self.state in ('packaged', 'updating'):
            return None
        if self.ctx.jobs.active_count():
            return 'Jobs are still running. Wait for them to finish, or cancel them, then update.'
        self.job = Job('Update ProTrack3D', ['update'], ['update'])
        self._set('updating')
        self.ctx.submit(self.job)
        return None

    def _job_finished(self, job):
        if job is not self.job:
            return
        self.job = None
        if job.state == 'done':
            result = job.results.get('update', {})
            if result.get('version'):
                self.info = dict(self.info, current=result['version'], latest=result['version'], available=False)
            self._set('updated' if result.get('updated') else 'current')
        elif job.state == 'cancelled':
            self._set('available')
        else:
            self._set('error', (job.error or job.message or 'The update failed.').split(': ', 1)[-1])

    def restart(self):
        """Start the app again and close this one. Returns a message when that is not possible now."""
        if self.ctx.jobs.active_count():
            return 'Jobs are still running. Wait for them to finish, or cancel them, then restart.'
        QProcess.startDetached(sys.executable, ['-W', 'ignore', '-m', 'protrack3d.app'], REPO_ROOT)
        QApplication.instance().quit()
        return None
