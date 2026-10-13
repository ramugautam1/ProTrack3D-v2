"""One colour per object id, shared by the viewer and the lineage view so an object looks the same in both."""
import numpy as np


def id_colours(n=4096):
    rng = np.random.default_rng(7)
    hue = rng.random(n)
    k = (np.stack([hue * 6 + 5, hue * 6 + 3, hue * 6 + 1], 1) % 6)
    rgb = 1 - 0.75 * np.clip(np.minimum(k, 4 - k), 0, 1)  # saturated, bright colours
    return (rgb * 255).astype(np.uint8)
