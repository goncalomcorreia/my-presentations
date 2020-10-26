import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

majorFormatter = FormatStrFormatter('%.1f')

font = {'family' : 'cmr10'}
plt.rc('font', **font)

t = np.arange(-2,2,.01)

def sparsemax_loss(t):
    if t >= 1: return 0
    if t <= -1: return -t
    else: return (t-1)**2/4

def softmax_loss(t):
    return np.log(1 + np.exp(-t))

def squared_loss(t):
    return t*t

def rotated_squared_loss(t):
    return (t-1)**2/4

def hinge_loss(t):
    if t >= 1: return 0
    else: return 1-t

def softplus_loss(t):
    if t >= 0: return 0
    else: return -t

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(majorFormatter)

#plt.plot(t, np.vectorize(softmax_loss)(t)/np.log(2), 'red', label='softmax ($\div \log(2)$)')
#plt.hold(True)
plt.plot(t, np.vectorize(softmax_loss)(t), 'blue', label='softmax', linewidth=3.0)
plt.hold(True)
plt.plot(t, np.vectorize(hinge_loss)(t), 'green', label='hinge', linewidth=3.0)
plt.hold(True)
#plt.plot(t, np.vectorize(squared_loss)(t), 'magenta', label='squared')
#plt.hold(True)
#plt.plot(t, np.vectorize(sparsemax_loss)(t)*4, 'blue', label='sparsemax ($\\times 4$)')
#plt.hold(True)
plt.plot(t, np.vectorize(rotated_squared_loss)(t), 'm--', label='least squares', linewidth=3.0)
plt.hold(True)
#plt.plot(t, np.vectorize(softplus_loss)(t), 'c--', label='softplus', linewidth=3.0)
#plt.hold(True)
plt.plot(t, np.vectorize(sparsemax_loss)(t), 'red', label='sparsemax', linewidth=3.0)
plt.hold(True)

plt.legend(loc=1, fontsize=20)
plt.xlabel('$t$', fontsize=16)
plt.ylim((-.1, 3))
plt.setp(plt.gca().get_xticklabels(), fontsize=16)        
plt.setp(plt.gca().get_yticklabels(), fontsize=16)        

plt.show()
