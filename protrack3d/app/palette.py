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


# --- colour tables (LUTs) for showing grey values, after ImageJ's Image > Lookup Tables ---

def _ramp(colour):
    return (np.linspace(0, 1, 256)[:, None] * np.array(colour, float)[None]).round().astype(np.uint8)


def _from_points(r, g, b):
    """A 256-entry table interpolated between evenly spaced control points, as ImageJ builds Fire."""
    x = np.linspace(0, 255, len(r))
    return np.stack([np.interp(np.arange(256), x, c) for c in (r, g, b)], 1).round().astype(np.uint8)


def _spectrum():
    hue = np.arange(256) / 255.0
    k = np.stack([(5 + hue * 6) % 6, (3 + hue * 6) % 6, (1 + hue * 6) % 6], 1)
    return ((1 - np.clip(np.minimum(k, 4 - k), 0, 1)) * 255).round().astype(np.uint8)


_SIXTEEN = [(0, 0, 0), (1, 1, 171), (1, 1, 224), (0, 110, 255), (1, 171, 254), (1, 224, 254), (1, 254, 1),
            (190, 255, 0), (255, 255, 0), (255, 224, 0), (255, 141, 0), (250, 94, 0), (245, 0, 0), (245, 0, 108),
            (222, 180, 222), (255, 255, 255)]

LUTS = {
    'Grays': _ramp((255, 255, 255)),
    'Red': _ramp((255, 0, 0)),
    'Green': _ramp((0, 255, 0)),
    'Blue': _ramp((0, 0, 255)),
    'Cyan': _ramp((0, 255, 255)),
    'Magenta': _ramp((255, 0, 255)),
    'Yellow': _ramp((255, 255, 0)),
    'Fire': _from_points(
        [0, 0, 1, 25, 49, 73, 98, 122, 146, 162, 173, 184, 195, 207, 217, 229, 240, 252, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 35, 57, 79, 101, 117, 133, 147, 161, 175, 190, 205, 219, 234, 248,
         255, 255, 255, 255],
        [0, 61, 96, 130, 165, 192, 220, 227, 210, 181, 151, 122, 93, 64, 35, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35,
         98, 160, 223, 255]),
    'Spectrum': _spectrum(),
    '16 colors': np.repeat(np.array(_SIXTEEN, np.uint8), 16, axis=0),
    'Inverted': _ramp((255, 255, 255))[::-1].copy(),
}
AUTO = 'Auto'  # grey when a channel is shown alone, its own colour when channels are mixed


def channel_lut(name, colour, alone):
    """The table a channel is drawn with: the one chosen for it, or for Auto, grey alone and its colour mixed."""
    if name != AUTO:
        return LUTS[name]
    return LUTS['Grays'] if alone else _ramp(colour)


def lut_end_colour(name, colour):
    """The colour that stands for a table on a swatch: its brightest entry, or the channel colour for Auto."""
    return tuple(int(v) for v in (colour if name == AUTO else LUTS[name][-1 if name != 'Inverted' else 0]))
