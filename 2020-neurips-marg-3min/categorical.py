import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

circle1 = plt.Circle((0.3, 0.5), 0.05, color='r')
circle2 = plt.Circle((0.5, 0.5), 0.05, color='blue')
circle3 = plt.Circle((0.7, 0.5), 0.05, color='g', clip_on=False)

fig, ax = plt.subplots(figsize=(10, 10))  # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

plt.axis('off')
plt.grid(False)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')

fig.savefig('plotcircles.png')

import tikzplotlib

tikzplotlib.save("test.tex")
