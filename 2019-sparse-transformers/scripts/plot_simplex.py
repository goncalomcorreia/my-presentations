import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import simplex
import projection

COLOR = "#a3a8a2"
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR

# plot linear over simplex

theta = np.array([0.5, 1, 0])
plt.figure(figsize=(4.5, 4))
proj = projection.Linear(theta)
# plt.tripcolor(simplex._triangle, facecolors=["#655638"], edgecolors=["#c3ac5c"])
plt.gca().fill(simplex._corners[:, 0], simplex._corners[:, 1], "#c3ac5c")
cs = simplex.draw(proj.val, nlevels=10)#
plt.clabel(cs,
           # cs.levels[1::3],
           rightside_up=True,
           inline=True,
           manual=[(.46, .6), (.4, .4), (.6, .18), (.75, .09)])
           #  manual=[(.5, t) for t in np.arange(0, 1, 0.2)])
opt = proj.project(verbose=True)
simplex.scatter(opt, marker='x', color='k')
plt.axis('off')
# plt.show()
plt.savefig("../img/4010_linear_simplex.pdf", bbox_inches="tight",
        transparent=True)
# plt.savefig("fig/0302_linear_simplex.pdf", bbox_inches="tight")


for k, theta in enumerate([np.zeros(3),
                           np.array([.8, 1, 0])]):
    for proj in (
                 projection.Entropy(theta),
                 projection.Squared(theta),
                 #  projection.PNormMax,
                 projection.Tsallis15(theta)
                 # projection.Oscarmax(theta, alpha=.2),
                 # projection.Fusedmax(theta, alpha=.2)
                 ):

        plt.figure(figsize=(4.5, 4))
        # plt.title("{} at {}".format(proj.title, theta))
        plt.gca().fill(simplex._corners[:, 0], simplex._corners[:, 1], "#c3ac5c")
        cs = simplex.draw(proj.val, nlevels=10, color="#c3ac5c")
        plt.clabel(cs,
                   # cs.levels[1::3],
                   rightside_up=True,
                   inline=True,
                   manual=[(.46, .6), (.4, .4), (.6, .18), (.75, .09)])
        opt = proj.project()
        print(opt)
        simplex.scatter(opt, marker='x', color='k')
        plt.savefig("../img/4011_simplex_{}_{}.pdf".format(k,
            proj.title), bbox_inches="tight", transparent=True)
        # plt.savefig("fig/0306_simplex_{}_{}.pdf".format(k, proj.title),
                    # bbox_inches="tight")
