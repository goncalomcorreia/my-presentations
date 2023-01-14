import re
import datetime
import numpy as np
import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt

log_re = re.compile("^\[(.+) INFO\] (.*)$")

matplotlib.rcParams['figure.constrained_layout.use'] = True


# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['pgf.preamble'] = [
    # r'\usepackage{times}',
    # r'\usepackage{amsmath}',
    # r'\usepackage{amssymb}',
    # r'\usepackage{bm}',
# #    r'\DeclareMathOperator*{\argmax}{\mathsf{argmax}}',
    # r'\DeclareMathOperator*{\softmax}{\mathsf{softmax}}',
    # r'\newcommand*\entmaxtext{entmax}',
    # r'\DeclareMathOperator*{\entmax}{\mathsf{\entmaxtext}}',
    # r'\newcommand*\aentmax[1][\alpha]{\mathop{\mathsf{#1}\textnormal{-}\mathsf{\entmaxtext}}}',
# ]
# matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 8
COLOR = "#a3a8a2"
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.verbose.level = 'debug-annoying'

def load_onmt_valid_timing(fname):

    with open(fname) as f:

        start_time = None
        times = []
        accs = []
        for line in f:
            match = log_re.match(line)

            if match:
                ts_s, msg = match.groups()
                ts = datetime.datetime.strptime(ts_s, '%Y-%m-%d %H:%M:%S,%f')

                if msg == "Start training...":
                    start_time = ts

                if msg.startswith("Validation accuracy"):
                    acc = float(msg.split()[-1])
                    times.append((ts - start_time).total_seconds())
                    accs.append(acc)

    return times, accs


def plot_both():
    so_so_1_time, so_so_1_acc = load_onmt_valid_timing('plot_logs/so-so-logs/run1-lr-001.log')
    so_so_2_time, so_so_2_acc = load_onmt_valid_timing('plot_logs/so-so-logs/run2-lr-001.log')
    so_so_3_time, so_so_3_acc = load_onmt_valid_timing('plot_logs/so-so-logs/run3-lr-001.log')

    sp_ts_1_time, sp_ts_1_acc = load_onmt_valid_timing('plot_logs/sp-ts-logs/run1-lr-001.log')
    sp_ts_2_time, sp_ts_2_acc = load_onmt_valid_timing('plot_logs/sp-ts-logs/run2-lr-001.log')
    sp_ts_3_time, sp_ts_3_acc = load_onmt_valid_timing('plot_logs/sp-ts-logs/run3-lr-001.log')

    # truncate length
    n_steps = np.min([
        len(so_so_1_acc),
        len(so_so_2_acc),
        len(so_so_3_acc),
        len(sp_ts_1_acc),
        len(sp_ts_2_acc),
        len(sp_ts_3_acc),
    ])

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,
        figsize=(9, 3),
#         subplot_kw=dict(xscale='log'),
    )


    kw = dict(alpha=1, color='C0', marker='o', ms=2)
    ax1.plot(range(1, n_steps + 1), so_so_1_acc[:n_steps], label="1-1", **kw)
    ax1.plot(range(1, n_steps + 1), so_so_2_acc[:n_steps], **kw)
    ax1.plot(range(1, n_steps + 1), so_so_3_acc[:n_steps], **kw)

    kw['color'] = 'C1'
    ax1.plot(range(1, n_steps + 1), sp_ts_1_acc[:n_steps], label="2-1.5", **kw)
    ax1.plot(range(1, n_steps + 1), sp_ts_2_acc[:n_steps], **kw)
    ax1.plot(range(1, n_steps + 1), sp_ts_3_acc[:n_steps], **kw)

    kw['color'] = 'C0'
    ax2.plot(so_so_1_time[:n_steps], so_so_1_acc[:n_steps], label="1-1", **kw)
    ax2.plot(so_so_2_time[:n_steps], so_so_2_acc[:n_steps], **kw)
    ax2.plot(so_so_3_time[:n_steps], so_so_3_acc[:n_steps], **kw)

    kw['color'] = 'C1'
    ax2.plot(sp_ts_1_time[:n_steps], sp_ts_1_acc[:n_steps], label="2-1.5", **kw)
    ax2.plot(sp_ts_2_time[:n_steps], sp_ts_2_acc[:n_steps], **kw)
    ax2.plot(sp_ts_3_time[:n_steps], sp_ts_3_acc[:n_steps], **kw)

    ax1.set_ylabel('validation accuracy')
    ax1.set_xlabel('checkpoints')
    ax2.set_xlabel('seconds')

    # ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.set_xlim(None, 17)
    ax1.yaxis.set_ticks((57.0, 58.5, 60, 61.5, 63))
    from matplotlib.ticker import FormatStrFormatter, PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter())

    # ax1.xaxis.set_major_locator(plt.LogLocator(base=2))
    # ax1.xaxis.set_minor_locator(plt.LogLocator(base=2))
    # ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    # ax1.xaxis.set_minor_formatter(plt.NullFormatter())

    # ax2.xaxis.set_major_locator(plt.LogLocator(base=2))
    # ax2.xaxis.set_minor_locator(plt.LogLocator(base=2))
    # ax2.xaxis.set_major_formatter(plt.ScalarFormatter())
    # ax2.xaxis.set_minor_formatter(plt.NullFormatter())

    plt.subplots_adjust(left=.15, bottom=.23, right=.98, top=.96, wspace=.06)


 #   ax2.plot(so_so_time, so_so_acc, marker='o', label="1-1")
 #   ax2.plot(sp_ts_time, sp_ts_acc, marker='o', label="2-1.5")

    plt.show()


def plot_one():
    so_so_1_time, so_so_1_acc = load_onmt_valid_timing('plot_logs/so-so-logs/run1-lr-001.log')
    so_so_2_time, so_so_2_acc = load_onmt_valid_timing('plot_logs/so-so-logs/run2-lr-001.log')
    so_so_3_time, so_so_3_acc = load_onmt_valid_timing('plot_logs/so-so-logs/run3-lr-001.log')

    sp_ts_1_time, sp_ts_1_acc = load_onmt_valid_timing('plot_logs/ts-ts-logs/run1-lr-001.log')
    sp_ts_2_time, sp_ts_2_acc = load_onmt_valid_timing('plot_logs/ts-ts-logs/run2-lr-001.log')
    sp_ts_3_time, sp_ts_3_acc = load_onmt_valid_timing('plot_logs/ts-ts-logs/run3-lr-001.log')

    # truncate length
    n_steps = np.min([
        len(so_so_1_acc),
        len(so_so_2_acc),
        len(so_so_3_acc),
        len(sp_ts_1_acc),
        len(sp_ts_2_acc),
        len(sp_ts_3_acc),
    ])

    f = plt.figure(figsize=(4, 1.8))
    ax = plt.gca()
    kw = dict(alpha=1, color='#00b5d6',
              #marker='o', ms=2,
              lw=1)

    ax.plot(so_so_1_time[:n_steps], so_so_1_acc[:n_steps], label="softmax", **kw)
    ax.plot(so_so_2_time[:n_steps], so_so_2_acc[:n_steps], **kw)
    ax.plot(so_so_3_time[:n_steps], so_so_3_acc[:n_steps], marker='|', ms=5, **kw)

    kw['color'] = '#e3627c'
    ax.plot(sp_ts_1_time[:n_steps], sp_ts_1_acc[:n_steps],
            label="1.5-entmax", **kw)
    ax.plot(sp_ts_2_time[:n_steps], sp_ts_2_acc[:n_steps],  **kw)
    ax.plot(sp_ts_3_time[:n_steps], sp_ts_3_acc[:n_steps],marker='|', ms=5,  **kw)

    legend = ax.legend(
       frameon=False,
       borderpad=.3,
       borderaxespad=0,
       handletextpad=0.5,
       fontsize=9,
       title_fontsize=7,
       labelspacing=.16,
       handlelength=1.2,
       fancybox=False)

    ax.set_ylabel('validation\naccuracy')
    ax.set_xlabel('seconds', labelpad=.2)

    # ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_ticks((57.0, 58.5, 60, 61.5, 63))
    from matplotlib.ticker import FormatStrFormatter, PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='y', which='major')
    ax.spines['bottom'].set_color(COLOR)
    ax.spines['left'].set_color(COLOR)
        # ax_proba.spines['top'].set_color('#dddddd')
        # ax_proba.spines['right'].set_color('red')

    # ax1.xaxis.set_major_locator(plt.LogLocator(base=2))
    # ax1.xaxis.set_minor_locator(plt.LogLocator(base=2))
    # ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    # ax1.xaxis.set_minor_formatter(plt.NullFormatter())

    # ax2.xaxis.set_major_locator(plt.LogLocator(base=2))
    # ax2.xaxis.set_minor_locator(plt.LogLocator(base=2))
    # ax2.xaxis.set_major_formatter(plt.ScalarFormatter())
    # ax2.xaxis.set_minor_formatter(plt.NullFormatter())

    # plt.subplots_adjust(left=.21, bottom=.29, right=.98, top=.9, wspace=.06)


 #   ax2.plot(so_so_time, so_so_acc, marker='o', label="1-1")
 #   ax2.plot(sp_ts_time, sp_ts_acc, marker='o', label="2-1.5")

    # plt.show()
    plt.savefig('../img/valid_curve.pdf', transparent=True)

if __name__ == '__main__':
    plot_one()







