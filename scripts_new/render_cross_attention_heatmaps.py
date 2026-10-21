#!/usr/bin/env python3
"""Render fixed-palette original/fused panels from trusted local heatmap files."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def render_case(heatmap_path, output_path):
    data = torch.load(heatmap_path, map_location='cpu')
    labels = [key for key in data if key != '__metadata__']
    with Image.open(data['__metadata__']['image_path']) as handle:
        image = handle.convert('RGB').resize((256, 256), Image.BICUBIC)
    canvas = Image.new('RGB', (256 * (len(labels) + 1), 280), 'white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image, (0, 24))
    draw.text((4, 4), 'Original: ' + data['__metadata__']['image_id'], fill='black')
    for index, label in enumerate(labels, 1):
        cam = data[label]['fused']['cam_vis_up'].detach().cpu()
        if cam.ndim != 2 or not bool(torch.isfinite(cam).all()) or bool((cam < 0).any() or (cam > 1).any()):
            raise ValueError('Expected finite normalized two-dimensional CAM')
        heat = np.asarray(Image.fromarray(cam.numpy(), mode='F').resize((256, 256), Image.BILINEAR))
        color = np.stack([255 * heat, np.zeros_like(heat), 255 * (1 - heat)], axis=-1).astype(np.uint8)
        overlay = Image.blend(image, Image.fromarray(color), .4)
        canvas.paste(overlay, (index * 256, 24))
        draw.text((index * 256 + 4, 4), label, fill='black')
    with Path(output_path).open('xb') as handle:
        canvas.save(handle, format='PNG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob('*.pt'))
    if not files:
        raise ValueError('No heatmap files found')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for path in files:
        render_case(path, args.output_dir/(path.stem + '.png'))
    print('Rendered', len(files), 'case panels to', args.output_dir)


if __name__ == '__main__':
    main()
