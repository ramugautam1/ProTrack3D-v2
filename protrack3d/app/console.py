"""Console: what the engine prints, inside the window. It can also run engine commands typed at the prompt."""
import html
import shlex

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLineEdit, QPlainTextEdit, QVBoxLayout

from . import icons
from .jobs import STAGE_LABELS, Job, format_seconds
from .widgets import button, text

HINT = 'Run an engine command:  models  ·  info image.tif  ·  segment image.tif --model NAME  ·  --help'


class Console(QFrame):
    closed = Signal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('console')
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(14, 6, 8, 6)
        head.addWidget(text('Console', 'consoleTitle'))
        self.status = text('', 'consoleStatus')
        head.addWidget(self.status)
        head.addStretch(1)
        self.buttons = []
        for label, tip, action in (('Copy', 'Copy everything', self.copy), ('Clear', 'Empty the console', self.clear), ('', 'Hide the console', self.closed.emit)):
            b = button(label, 'consoleButton', tip=tip)
            b.clicked.connect(action)
            head.addWidget(b)
            self.buttons.append(b)
        box.addLayout(head)
        self.view = QPlainTextEdit()
        self.view.setObjectName('terminal')
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(6000)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        box.addWidget(self.view, 1)
        line = QHBoxLayout()
        line.setContentsMargins(14, 0, 8, 0)
        line.setSpacing(6)
        line.addWidget(text('protrack3d ›', 'consolePrompt'))
        self.input = QLineEdit()
        self.input.setObjectName('consoleInput')
        self.input.setPlaceholderText(HINT)
        self.input.returnPressed.connect(self.run_command)
        line.addWidget(self.input, 1)
        box.addLayout(line)
        self.history, self.at = [], 0
        self.input.installEventFilter(self)

        jobs = ctx.jobs
        jobs.started.connect(self.job_started)
        jobs.output.connect(lambda job, line: self.write(line))
        jobs.event.connect(self.job_event)
        jobs.finished.connect(self.job_finished)
        ctx.theme.listeners.append(self.restyle)
        self.restyle()

    def restyle(self):
        self.buttons[-1].setIcon(icons.icon('x', '#9BA8B6', 14))

    # --- writing ---
    def write(self, line, colour=None, bold=False):
        bar = self.view.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        safe = html.escape(line).replace(' ', '&nbsp;')
        style = (f'color:{colour};' if colour else '') + ('font-weight:600;' if bold else '')
        self.view.appendHtml(f'<span style="{style}">{safe}</span>' if style else safe)
        if follow:
            bar.setValue(bar.maximum())

    def job_started(self, job):
        c = self.ctx.theme.colors
        self.write('')
        self.write(f'▶ {job.title}', '#FFFFFF', bold=True)
        self.write('$ protrack3d ' + ' '.join(shlex.quote(a) for a in job.args), '#8B97A5')
        self.status.setText('  running')

    def job_event(self, job, event):
        kind = event.get('event')
        stage = STAGE_LABELS.get(event.get('stage'), event.get('stage') or '')
        if kind == 'stage':
            self.write(f'── {stage}: {event.get("message", "")}', '#35C6AE', bold=True)
        elif kind == 'progress':
            count = f'{event["done"]}/{event["total"]}  ' if event.get('total') else ''
            left = f'   (about {format_seconds(event["eta"])} left)' if (event.get('eta') or 0) > 0 else ''
            self.write(f'   {count}{event.get("message", "")}{left}', '#9BA8B6')
        elif kind == 'epoch':
            self.write(f"   epoch {event['epoch']}/{event['epochs']}: loss {event['loss']:.4f}, validation IoU {event['validation_iou']:.4f} "
                       f"(best {event['best_iou']:.4f} at epoch {event['best_epoch']})", '#E7ECF1')
        elif kind == 'stage_done':
            self.write(f'✓ {stage} finished in {format_seconds(event.get("elapsed", 0))}', '#4CC38A')
        elif kind == 'error':
            self.write(f'✗ {event.get("message", "")}', '#F0716A', bold=True)

    def job_finished(self, job):
        colour = {'done': '#4CC38A', 'failed': '#F0716A'}.get(job.state, '#E0A341')
        words = {'done': 'finished', 'failed': 'failed', 'cancelled': 'cancelled'}
        self.write(f'■ {job.title}: {words.get(job.state, job.state)} after {format_seconds(job.elapsed())}', colour, bold=True)
        self.status.setText('  running' if self.ctx.jobs.active_count() else '')

    def copy(self):
        QGuiApplication.clipboard().setText(self.view.toPlainText())
        self.ctx.toast('Console copied')

    def clear(self):
        self.view.clear()

    # --- the prompt ---
    def run_command(self):
        line = self.input.text().strip()
        if not line:
            return
        self.history.append(line)
        self.at = len(self.history)
        self.input.clear()
        try:
            args = shlex.split(line)
        except ValueError as e:
            self.write(f'✗ {e}', '#F0716A')
            return
        if args and args[0] in ('protrack3d', 'python'):
            args = args[args.index('protrack3d') + 1:] if 'protrack3d' in args else args[1:]
        self.ctx.submit(Job(f'Command  ·  {line}', args, []), announce=False)

    def eventFilter(self, watched, event):
        if watched is self.input and event.type() == event.Type.KeyPress and self.history:
            if event.key() in (Qt.Key_Up, Qt.Key_Down):
                self.at = max(0, min(len(self.history), self.at + (-1 if event.key() == Qt.Key_Up else 1)))
                self.input.setText(self.history[self.at] if self.at < len(self.history) else '')
                return True
        return super().eventFilter(watched, event)
