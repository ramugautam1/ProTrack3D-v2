"""Updating ProTrack3D in place, for a copy that runs from the source (a git clone, or the ZIP from GitHub).

A git clone is brought up to date with git. A ZIP download has no history, so its files are compared with the
published ones and the ones that differ are replaced from a fresh download. Models, results and anything else
that is not part of the published code are left alone. The packaged app cannot update itself.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

from .events import Reporter
from .paths import FROZEN, REPO_ROOT

REPO = os.environ.get('PROTRACK3D_UPDATE_REPO') or 'ramugautam1/ProTrack3D-v2'
BRANCH = 'main'
API = f'https://api.github.com/repos/{REPO}'
STATE = os.path.join(REPO_ROOT, '.protrack3d_update.json')  # what a ZIP copy was last updated to
PENDING = os.path.join(REPO_ROOT, '.protrack3d_update_pending')  # new packages still have to be installed
REQUIREMENTS = os.path.join(REPO_ROOT, 'requirements.txt')


def kind():
    """How this copy was installed: 'packaged', 'git' or 'zip'."""
    if FROZEN:
        return 'packaged'
    if os.path.exists(os.path.join(REPO_ROOT, '.git')) and shutil.which('git'):
        return 'git'
    return 'zip'


def _git(*args):
    done = subprocess.run(['git', '-C', REPO_ROOT, *args], capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT='0'))
    if done.returncode:
        raise RuntimeError((done.stderr or done.stdout).strip() or f'git {args[0]} failed')
    return done.stdout.strip()


def _open(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ProTrack3D', 'Accept': 'application/vnd.github+json'})
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise RuntimeError('GitHub is not answering right now (too many requests from this network). Try again in an hour.') from None
        raise RuntimeError(f'GitHub answered with an error ({e.code}) for {url}') from None
    except OSError as e:
        raise RuntimeError(f'could not reach GitHub: {getattr(e, "reason", e)}') from None


def _json(url):
    with _open(url) as response:
        return json.load(response)


def _blob_id(data):
    """The name git gives to a file with this content."""
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def _local_id(relative):
    try:
        with open(os.path.join(REPO_ROOT, *relative.split('/')), 'rb') as f:
            return _blob_id(f.read())
    except OSError:
        return None


def _read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _published():
    """The newest published commit and the files it holds, as {path: git id}."""
    commit = _json(f'{API}/commits/{BRANCH}')
    tree = _json(f'{API}/git/trees/{commit["commit"]["tree"]["sha"]}?recursive=1')
    files = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
    latest = {'id': commit['sha'][:7], 'date': commit['commit']['committer']['date'][:10],
              'title': commit['commit']['message'].splitlines()[0], 'commit': commit['sha']}
    return latest, files


def _git_news():
    _git('fetch', '--quiet', 'origin', BRANCH)
    return [line for line in _git('log', '--format=%s', 'HEAD..FETCH_HEAD').splitlines() if line]


def _git_version(ref):
    ident, date = _git('log', '-1', '--format=%h|%cs', ref).split('|')
    return {'id': ident, 'date': date}


def check():
    """Is a newer version published? Returns a dict; 'available' is True when updating would change something."""
    how = kind()
    if how == 'packaged':
        return {'kind': how, 'available': False, 'changes': [], 'current': None, 'latest': None}
    if how == 'git':
        news = _git_news()
        result = {'kind': how, 'available': bool(news), 'changes': news, 'files': None,
                  'current': _git_version('HEAD'), 'latest': _git_version('FETCH_HEAD')}
    else:
        latest, files = _published()
        changed = [path for path, ident in files.items() if _local_id(path) != ident]
        state = _read_state()
        current = {'id': state['commit'][:7], 'date': state.get('date')} if state.get('commit') else None
        result = {'kind': how, 'available': bool(changed), 'changes': [latest['title']] if changed else [],
                  'files': len(changed), 'current': current, 'latest': {k: latest[k] for k in ('id', 'date')}}
        if not changed:
            result['current'] = result['latest']
    if os.path.isfile(PENDING):
        result['available'] = True
        result['changes'] = result['changes'] or ['Finish installing the new packages']
    return result


def _update_from_zip(reporter):
    latest, files = _published()
    changed = [path for path, ident in files.items() if _local_id(path) != ident]
    if not changed:
        return 0, latest
    work = tempfile.mkdtemp(prefix='protrack3d-update-')
    try:
        archive = os.path.join(work, 'new.zip')
        reporter.progress(10, 100, 'downloading the new version')
        with _open(f'https://github.com/{REPO}/archive/{latest["commit"]}.zip') as response, open(archive, 'wb') as out:
            got = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                reporter.progress(10, 100, f'downloading the new version: {got / 1e6:.1f} MB')
        new = {}
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            top = members[0].filename.split('/')[0] + '/'
            for member in members:
                relative = member.filename[len(top):]
                if not member.filename.startswith(top) or not relative or relative.startswith('/') or '..' in relative.split('/'):
                    raise RuntimeError('the download holds a file where none was expected; nothing was changed.')
                new[relative] = (z.read(member), member.external_attr >> 16)
        # Nothing is written unless every file arrived exactly as published.
        if set(files) - set(new) or any(_blob_id(new[path][0]) != ident for path, ident in files.items()):
            raise RuntimeError('the download arrived damaged (it does not match the published files); nothing was changed.')
        reporter.progress(50, 100, f'replacing {len(changed)} file(s)')
        for relative in changed:
            data, mode = new[relative]
            target = os.path.join(REPO_ROOT, *relative.split('/'))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target + '.new', 'wb') as f:
                f.write(data)
            if mode & 0o111:  # the launchers must stay runnable
                os.chmod(target + '.new', 0o755)
            os.replace(target + '.new', target)
        for relative in set(_read_state().get('files', [])) - set(files):  # files the new version no longer has
            stale = os.path.join(REPO_ROOT, *relative.split('/'))
            if os.path.isfile(stale):
                os.remove(stale)
        with open(STATE, 'w') as f:
            json.dump({'commit': latest['commit'], 'date': latest['date'], 'files': sorted(files)}, f)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return len(changed), latest


def _file_id(path):
    try:
        with open(path, 'rb') as f:
            return _blob_id(f.read())
    except OSError:
        return None


def _install_packages(reporter):
    open(PENDING, 'w').close()
    reporter.progress(70, 100, 'installing new packages')
    process = subprocess.Popen([sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        reporter.stream.write(line)
        reporter.stream.flush()
    if process.wait():
        raise RuntimeError('the files were updated, but the new packages could not be installed (see the log). '
                           'Check the internet connection and press Update again.')
    os.remove(PENDING)


def apply(reporter=None):
    """Bring this copy up to the newest published version. The app has to be started again afterwards."""
    reporter = reporter or Reporter()
    how = kind()
    if how == 'packaged':
        raise RuntimeError('the packaged app cannot update itself. Download the new version instead.')
    reporter.stage('update', total=100, message='looking for a newer version')
    packages_before = _file_id(REQUIREMENTS)
    if how == 'git':
        news = _git_news()
        if news:
            reporter.progress(40, 100, f'getting {len(news)} change(s)')
            try:
                _git('merge', '--ff-only', 'FETCH_HEAD')
            except RuntimeError as e:
                raise RuntimeError('git could not update this folder. Files in it were changed by hand, or it has '
                                   f'commits of its own:\n{e}') from None
        changed, version = len(news), _git_version('HEAD')
    else:
        changed, latest = _update_from_zip(reporter)
        version = {k: latest[k] for k in ('id', 'date')}
    if os.path.isfile(PENDING) or _file_id(REQUIREMENTS) != packages_before:
        _install_packages(reporter)
        changed = changed or 1
    reporter.progress(100, 100, 'done' if changed else 'already up to date')
    result = {'updated': bool(changed), 'changes': changed, 'version': version, 'kind': how}
    reporter.stage_done(**result)
    return result
