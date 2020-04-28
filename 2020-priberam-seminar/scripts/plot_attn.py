#!/usr/bin/env python

import argparse
from os.path import basename, join
from itertools import repeat
import torch
import numpy as np
import matplotlib

# matplotlib.use('pgf')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.colors as colors

# matplotlib.use('pgf')

# matplotlib.rcParams['figure.constrained_layout.use'] = True


'''
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
'''
COLOR = "#a3a8a2"
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR


# matplotlib.rcParams['text.latex.preamble'] = [
    # r'\usepackage[T1]{fontenc}',
    # r'\usepackage{times}',
    # r'\usepackage{amsmath}',
    # r'\usepackage{amssymb}',
# ]


# matplotlib.rcParams['font.family'] = 'serif'


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either
    side from a prescribed midpoint value)

    e.g.
    im=ax1.imshow(arr, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def config(path):
    return tuple(basename(path).split('-')[:2])


def draw_square(ax, i, j, **kwargs):
    rect = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, **kwargs)
    ax.add_patch(rect)


def draw_all_squares(ax, M):
    for ii in range(M.shape[0]):
        for jj in range(M.shape[1]):
            if M[ii, jj] > 0:
                draw_square(ax, jj, ii, color="#aaaaaa", lw=1, alpha=1)


def draw_plot(plots, path, fontsize=10, rotation=45, size=2.5):
    if isinstance(plots, dict):
        plots = plots,

    plots = plots[1],

    fig = plt.figure(figsize=(size, size))

    for i, plot in enumerate(plots, 1):
        src = plot["src"] + plot["inflection"]  # should be same for both
        pred = plot["pred"]
        x_labels = [''] + list(src)
        y_labels = [''] + list(pred)

        if "gate" in plot["attn"]:
            lemma_gate = plot["attn"]["gate"][:, 0].unsqueeze(1)
            lemma_plot = lemma_gate * plot["attn"]["lemma"]

            inflection_gate = plot["attn"]["gate"][:, 1].unsqueeze(1)
            inflection_plot = inflection_gate * plot["attn"]["inflection"]
        else:
            lemma_plot = plot["attn"]["lemma"]
            inflection_plot = plot["attn"]["inflection"]

        attn_matrix = torch.cat([lemma_plot, inflection_plot], dim=1)

        ax = fig.add_subplot(1, len(plots), i)
        cmap = plt.cm.PuOr_r  # OrRd
        cax = ax.matshow(
            attn_matrix, cmap=cmap, clim=(-1, 1),
            norm=MidpointNormalize(midpoint=0,vmin=1, vmax=1)
        )
        draw_all_squares(ax, attn_matrix)
        ax.axvline(x=len(plot['src']) - 0.5, linestyle='--', color="black")
        # fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels(
            x_labels,
            rotation=rotation,
            fontsize=fontsize,
            horizontalalignment='left'
        )
        ax.set_yticklabels(y_labels, fontsize=fontsize)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        if i == 1:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        else:
            ax.yaxis.set_major_locator(ticker.NullLocator())

    plt.savefig(path, bbox_inches='tight', transparent=True)
    plt.close()


def usable_plots(*plots, reference=None):
    """
    Checks that all plots generate the same prediction. If reference is not
    None, the plots must predict it.
    """
    if reference is None:
        reference = plots[0]["pred"]
    return all(p["pred"] == reference for p in plots)


def load_reference(path):
    with open(path) as f:
        return [list(line.split('\t')[1]) + ["</s>"] for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("-double", type=torch.load)
    parser.add_argument("-gate", type=torch.load)
    parser.add_argument("-reference", type=load_reference, default=repeat(None))
    parser.add_argument('-fontsize', type=int, default=10)
    parser.add_argument("-rotation", type=int, default=45)
    parser.add_argument("-size", type=float, default=2.5)
    opt = parser.parse_args()
    assert opt.double is not None and opt.gate is not None

    plots = zip(opt.double, opt.gate)

    for i, (plot, gold) in enumerate(zip(plots, opt.reference), 1):
        if i != 83:
            continue
        if usable_plots(*plot, reference=gold):
            out_path = join(opt.out_dir, "plot{}.pdf".format(i))
            draw_plot(
                plot, out_path, fontsize=opt.fontsize, rotation=opt.rotation,
                size=opt.size
            )


if __name__ == '__main__':
    main()
