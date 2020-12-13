import numpy as np

import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import cm

from bisection import optimize_bisect, TsallisSeparable
from scipy.special import softmax


def project(t, alpha):
    x = np.array([t, 0])
    if alpha == 1:
        p = softmax(x)
    else:
        penalty = TsallisSeparable(alpha)
        p = optimize_bisect(x, penalty)[0]

    return p[0]


DRAFT = False
FIXED_RANGE = True
ALL = False

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['pgf.preamble'] = [
    # # r'\usepackage{times}',
    # # r'\usepackage{amsmath}',
    # # r'\usepackage{amssymb}',
    # # r'\usepackage{bm}',
# #    r'\DeclareMathOperator*{\argmax}{\mathsf{argmax}}',
# #    r'\DeclareMathOperator*{\softmax}{\mathsf{softmax}}',
# #    r'\DeclareMathOperator*{\sparsemax}{\mathsf{sparsemax}}',
# ]
# matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 6
#matplotlib.verbose.level = 'debug-annoying'

colors = {
    'softmax': "#00b5d6",
    'sparsemax': "#ddd9b8",
    # 'entmax15': plt.cm.Set3.colors[3],
    'entmax15': '#e3627c',
    'huge': '#777f75',
    'entmax125': plt.cm.Set3.colors[0],
}
# colors = {
    # 'softmax': plt.cm.Set3.colors[2],
    # 'sparsemax': plt.cm.Set3.colors[0],
    # # 'entmax15': plt.cm.Set3.colors[3],
    # 'entmax15': 'k',
    # 'entmax175': plt.cm.Set3.colors[3],
    # 'huge': 'gray'
# }

COLOR = "#a3a8a2"
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR

# colors = plt.cm.tab10.colors

tsallis_args = [
    (1.00001, dict(label=r'$\alpha=1$ (softmax)',
                   lw=2,
                   ls=":",
                   color=colors['softmax'],
                   zorder=10)),
    (1.25, dict(label=r'$\alpha=1.25$',
               lw=1,
               ls='-',
               color=colors['entmax125'],
               zorder=40)),
    (1.5, dict(label=r'$\alpha=1.5$',
               lw=1.5,
               # ls='--',
               color=colors['entmax15'],
               zorder=50)),
    (2, dict(label=r'$\alpha=2$ (sparsemax)',
             lw=2,
             alpha=1,  # 0.75,
             ls='--',
             color=colors['sparsemax'],
             zorder=30)),
    (4, dict(label=r'$\alpha=4$',
             lw=.5,
             ls='-',
             color=colors['huge'],
             zorder=20)),
#    (1000, dict(label=r'$\alpha=\infty$ (argmax)',
#                lw=1,
#                ls=':',
#                color='gray')),
]

def plot_entropy(ax_proba, args, title):

    n = 10 if DRAFT else 1000

    ts = np.linspace(-3.8, 3.8, n)

    for alpha, plot_args in args:
        #omega_ = omega(q=q)
        #omega_conj_ = omega_conjugate(q, omega=omega)
        #omega_values = [-omega_(p) for p in ps]

        # probability mapping
        pstar = [project(t, alpha) for t in ts]
        line, = ax_proba.plot(ts, pstar, **plot_args)
        line.set_solid_joinstyle('miter')
        ax_proba.set_xlabel('t', labelpad=0.5)
        # ax_proba.set_ylabel()

        legend = ax_proba.legend(title=title,
                                frameon=False,
                                borderpad=.3,
                                borderaxespad=0,
                                handletextpad=0.5,
                                fontsize=6,
                                title_fontsize=7,
                                labelspacing=.16,
                                handlelength=1.2,
                                fancybox=False)

        ax_proba.spines['bottom'].set_color(COLOR)
        # ax_proba.spines['top'].set_color('#dddddd')
        # ax_proba.spines['right'].set_color('red')
        ax_proba.spines['left'].set_color(COLOR)


def fig_acl():
    f = plt.figure(figsize=(3, 1.75))
    ax = f.gca()
    # ax.set_title(r'Predictive distribution $\alpha$-entmax$([s, 0])_1$')

    colors = plt.cm.tab10.colors

    plot_entropy(ax, tsallis_args, title=None)
    # left bottom right top wspace hspace
    # plt.subplots_adjust(.035, .13, .999, .93, .15, .15)
    plt.subplots_adjust(.11, .2, .999, .99, .15, .15)
    #plt.savefig('../img/entmax_mappings.pdf', transparent=True)
    import tikzplotlib
    tikzplotlib.save("../entmax_mappings.tex")


if __name__ == '__main__':
    fig_acl()
