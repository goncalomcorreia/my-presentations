import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

matplotlib.rcParams['font.size'] = 9

COLOR = "#a3a8a2"
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.verbose.level = 'debug-annoying'

def collect(fname):

    alphas = []
    accs = []
    for line in open(fname):
        alpha, acc, _ = line.strip().split()

        alpha = float(alpha)
        acc = float(acc)

        alphas.append(alpha)
        accs.append(acc)

    return alphas, accs


if __name__ == '__main__':



    # which = word = "loss"
            #"alpha_{}_results.txt".format(which)):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3, 3),
            constrained_layout=True)

    major_locator = FixedLocator([1, 1.25, 1.5, 1.75, 2, 2.25])
    minor_locator = FixedLocator([1.125, 1.375, 1.625, 1.875, 2.125])

    alphas_attn, accs_attn = collect("alpha_attn_results.txt")
    alphas_loss, accs_loss = collect("alpha_loss_results.txt")
    ax1.plot(alphas_attn, accs_attn, marker='o', color=COLOR, ms=4)
    ax2.plot(alphas_loss, accs_loss, marker='o', color=COLOR, ms=4)
    #ax2.set_ylim((61, 63.75))
    ax1.set_ylabel("validation\naccuracy")#, labelpad=.5)
    ax2.set_ylabel("validation\naccuracy")#, labelpad=.5)
    ax1.set_xlabel("attention $\\alpha$")#, labelpad=.2)
    ax2.set_xlabel("output $\\alpha$")#, labelpad=.2)

    ax1.yaxis.set_ticks((62, 63, 64))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d%%'))
    ax1.tick_params(axis='y', which='major')

    # from matplotlib.ticker import PercentFormatter
    ax2.yaxis.set_ticks((60, 61, 62, 63))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d%%'))
    # ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax2.tick_params(axis='y', which='major')

    for ax in (ax1, ax2):
        ax.spines['bottom'].set_color(COLOR)
        ax.spines['left'].set_color(COLOR)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)

    # plt.subplots_adjust(left=.1, bottom=.3, right=1, top=1, wspace=0.05)
    plt.savefig('../img/alpha_grid.pdf', transparent=True)
    # plt.show()

