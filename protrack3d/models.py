"""Model files. A model is `<name>.pt` (weights + metadata) with a `<name>.json` copy of the metadata beside it,
so the models can be listed without loading the weights."""
import glob
import json
import os

ARCHITECTURE = 'FC-DenseNet103'


def pick_device(name=None):
    import torch
    name = name if name and name != 'auto' else os.environ.get('PROTRACK3D_DEVICE')  # e.g. PROTRACK3D_DEVICE=cpu
    if name and name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_model(path, model, meta):
    import torch
    meta = dict(meta, architecture=ARCHITECTURE, name=os.path.splitext(os.path.basename(path))[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, path)
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def load_model(path, device=None):
    import torch
    from .network import FCDenseNet103
    data = torch.load(path, map_location='cpu', weights_only=True)
    model = FCDenseNet103()
    model.load_state_dict(data['state_dict'])
    return model.eval().to(device or pick_device()), data['meta']


def list_models(folder):
    models = []
    for path in sorted(glob.glob(os.path.join(folder, '*.pt'))):
        meta = {'name': os.path.splitext(os.path.basename(path))[0]}
        sidecar = os.path.splitext(path)[0] + '.json'
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                meta.update(json.load(f))
        meta['path'] = path
        meta['size_mb'] = round(os.path.getsize(path) / 1e6, 1)
        models.append(meta)
    return models
