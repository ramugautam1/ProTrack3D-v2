"""Progress reporting. With json_mode the engine prints one JSON object per line, prefixed with MARKER,
so a front end (the app, or the Claude skill) can tell progress events apart from ordinary log output."""
import json
import sys
import time

MARKER = '@@PT3D '


class Reporter:
    def __init__(self, json_mode=False, stream=None):
        self.json_mode = json_mode
        self.stream = stream or sys.__stdout__
        self._stage = None
        self._stage_start = None

    def _emit(self, **event):
        if self.json_mode:
            # start on a fresh line: the tracking code prints progress without a trailing newline
            self.stream.write('\n' + MARKER + json.dumps(event) + '\n')
            self.stream.flush()

    def stage(self, name, total=None, message=''):
        self._stage, self._stage_start = name, time.time()
        self._emit(event='stage', stage=name, total=total, message=message)
        if not self.json_mode:
            self.stream.write(f'\n== {name}: {message}\n')
            self.stream.flush()

    def progress(self, done, total, message=''):
        elapsed = time.time() - (self._stage_start or time.time())
        # time left only makes sense when the amount of work is known and not yet exceeded
        eta = elapsed / done * (total - done) if done and total and done <= total else None
        self._emit(event='progress', stage=self._stage, done=done, total=total, message=message,
                   elapsed=round(elapsed, 1), eta=None if eta is None else round(eta, 1))
        if not self.json_mode:
            count = f'{done}/{total} ' if total else ''
            self.stream.write(f'\r{self._stage}: {count}{message}   ')
            self.stream.flush()

    def epoch(self, **info):
        self._emit(event='epoch', stage=self._stage, **info)
        if not self.json_mode:
            self.stream.write(f"\nepoch {info['epoch']}/{info['epochs']}: loss {info['loss']:.4f}, validation IoU "
                              f"{info['validation_iou']:.4f} (best {info['best_iou']:.4f} at epoch {info['best_epoch']})\n")
            self.stream.flush()

    def stage_done(self, **info):
        elapsed = round(time.time() - (self._stage_start or time.time()), 1)
        self._emit(event='stage_done', stage=self._stage, elapsed=elapsed, **info)
        if not self.json_mode:
            self.stream.write(f'\n{self._stage} finished in {elapsed} s\n')
            self.stream.flush()

    def result(self, **info):
        self._emit(event='result', **info)
        if not self.json_mode:
            for key, value in info.items():
                self.stream.write(f'{key}: {value}\n')
            self.stream.flush()

    def error(self, message, stage=None):
        self._emit(event='error', stage=stage or self._stage, message=message)
        if not self.json_mode:
            where = stage or self._stage
            self.stream.write(f'\nERROR{f" ({where})" if where else ""}: {message}\n')
            self.stream.flush()
