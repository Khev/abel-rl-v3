#!/usr/bin/env python3
"""Compose a multi-panel t-SNE figure from saved per-dataset PNGs.

Just stitches 3 already-generated t-SNE PNGs side-by-side with a single
shared title. Used for the paper's appendix to compare cluster structure
across datasets.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PANELS = [
    ("abel_level3 (small)", "figures/tsne_l3_rcbuf.png"),
    ("abel_level4 (large)", "figures/tsne_l4_rcbuf.png"),
    ("poesia-full",         "figures/tsne_poesia_rc.png"),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, path) in zip(axes, PANELS):
        if not Path(path).exists():
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"missing\n{path}", ha="center", va="center", transform=ax.transAxes)
            continue
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
    fig.suptitle("TreeMLP pooled embeddings (t-SNE), colored by equation type",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = "figures/tsne_panel.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
