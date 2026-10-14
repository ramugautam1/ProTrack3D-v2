"""One colour per object id, shared by the viewer and the lineage view so an object looks the same in both."""
import numpy as np


def id_colours(n=4096):
    rng = np.random.default_rng(7)
    hue = rng.random(n)
    k = (np.stack([hue * 6 + 5, hue * 6 + 3, hue * 6 + 1], 1) % 6)
    rgb = 1 - 0.75 * np.clip(np.minimum(k, 4 - k), 0, 1)  # saturated, bright colours
    return (rgb * 255).astype(np.uint8)


DEFAULT_CHANNEL_COLOURS = [(0, 255, 0), (255, 0, 255), (0, 220, 255), (255, 60, 40), (255, 220, 0), (60, 110, 255)]


def channel_colour(name, index):
    """A display colour for a channel: from the wavelength in its name when there is one (Alexa Fluor 488 is green,
    568 red, 647 magenta), otherwise from a fixed list, as ImageJ's composite view does."""
    import re
    for number in re.findall(r'\d{3}', name or ''):
        nm = int(number)
        if 400 <= nm <= 470:
            return (60, 110, 255)
        if 470 < nm <= 540:
            return (0, 255, 0)
        if 540 < nm <= 620:
            return (255, 60, 40)
        if 620 < nm <= 780:
            return (255, 0, 255)
    return DEFAULT_CHANNEL_COLOURS[index % len(DEFAULT_CHANNEL_COLOURS)]
