"""The published models: what can be downloaded, and downloading it.

model_catalog.json lists each model with its size, checksum and description. The files themselves are too large for the
repository; they are fetched from `base_url` (set PROTRACK3D_MODEL_URL to use another address).
"""
import hashlib
import json
import os
import urllib.request

from .events import Reporter
from .paths import models_dir

CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_catalog.json')


def load_catalog():
    with open(CATALOG) as f:
        catalog = json.load(f)
    catalog['base_url'] = os.environ.get('PROTRACK3D_MODEL_URL') or catalog['base_url']
    for model in catalog['models']:
        model['installed'] = os.path.isfile(os.path.join(models_dir(), model['file']))
    return catalog


def download(names=None, reporter=None):
    """Download the named models (default: every model not installed yet) into the models folder."""
    reporter = reporter or Reporter()
    catalog = load_catalog()
    known = {m['name']: m for m in catalog['models']}
    unknown = [n for n in (names or []) if n not in known]
    if unknown:
        raise ValueError(f'not in the catalog: {", ".join(unknown)}. Available: {", ".join(known)}')
    wanted = [known[n] for n in names] if names else [m for m in catalog['models'] if not m['installed']]
    total = sum(m['size'] for m in wanted)
    os.makedirs(models_dir(), exist_ok=True)
    reporter.stage('download', total=100, message=f'{len(wanted)} model(s), {total / 1e6:.0f} MB')
    done, fetched = 0, []
    for model in wanted:
        target = os.path.join(models_dir(), model['file'])
        part, digest = target + '.part', hashlib.sha256()
        url = catalog['base_url'].rstrip('/') + '/' + model['file']
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, 'wb') as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    reporter.progress(min(99, int(100 * done / max(1, total))), 100, f'{model["name"]}: {done / 1e6:.0f} of {total / 1e6:.0f} MB')
        except Exception as e:
            if os.path.isfile(part):
                os.remove(part)
            raise RuntimeError(f'could not download {model["name"]} from {url}: {e}') from None
        if digest.hexdigest() != model['sha256']:
            os.remove(part)
            raise RuntimeError(f'{model["name"]} arrived damaged (its checksum does not match); it was not installed.')
        os.replace(part, target)
        with open(os.path.splitext(target)[0] + '.json', 'w') as f:
            json.dump(model['meta'], f, indent=2)
        fetched.append(model['name'])
    reporter.progress(100, 100, 'done')
    result = {'models': fetched, 'folder': models_dir()}
    reporter.stage_done(**result)
    return result
