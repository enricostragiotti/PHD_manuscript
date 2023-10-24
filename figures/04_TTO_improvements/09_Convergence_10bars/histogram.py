import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
from itertools import product, combinations
import glob
import imageio
import os
import tikzplotlib

# to change default colormap
plt.rcParams["image.cmap"] = "Set1"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

sample = 8
norm = matplotlib.colors.Normalize(vmin=0, vmax=sample-1) #normalize item number values to colormap
cmap = matplotlib.cm.get_cmap('coolwarm')

c_0_n = cmap(norm(7))
c_0 = cmap(norm(6))
c_1 = cmap(norm(1))
c_4 = cmap(norm(0))

data_0_n = np.load('0_n.npy')[1:50]
data_0 = np.load('0.npy')[1:50]
data_1 = np.load('1.npy')[1:50]
data_4 = np.load('4.npy')[1:50]

data_0_n_s = np.array(np.load('0_n_stolpe.npy')).reshape(-1)
data_0_s = np.array(np.load('0_stolpe.npy')).reshape(-1)
data_1_s = np.array(np.load('1_stolpe.npy')).reshape(-1)
data_4_s = np.array(np.load('4_stolpe.npy')).reshape(-1)

data_0_n = np.concatenate([data_0_n_s,data_0_n])
data_0 = np.concatenate([data_0_s,data_0])
data_1 = np.concatenate([data_1_s,data_1])
data_4 = np.concatenate([data_4_s,data_4])


mm = 1/25.4  # mm in inches
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
fig, ax1 = plt.subplots(figsize=(100*mm,80*mm))
#fig, ax1 = plt.subplots(figsize=(6,4))
ax1.grid(True)
# Calculate statistical measure of interest
mean_0_n = np.mean(data_0_n)
s_0_n = np.std(data_0_n)
mean_0 = np.mean(data_0)
s_0 = np.std(data_0)
mean_1 = np.mean(data_1)
s_1 = np.std(data_1)
mean_4 = np.mean(data_4)
s_4 = np.std(data_4)

# Plot mean
ax1.axhline(y = mean_0_n, color = c_0_n, xmin=-5, xmax = 5, linewidth=0.5, linestyle = 'dashed', zorder = 1)
ax1.axhline(y = mean_0, color = c_0, linewidth=0.5, linestyle = 'dashed', zorder = 1)
ax1.axhline(y = mean_1, color = c_1, linewidth=0.5, linestyle = 'dashed', zorder = 1)
ax1.axhline(y = mean_4, color = c_4, linewidth=0.5, linestyle = 'dashed', zorder = 1)

trans = transforms.blended_transform_factory(
    ax1.get_yticklabels()[0].get_transform(), ax1.transData)
ax1.text(0,mean_0_n, "{:.3f}".format(mean_0_n/1e5), color=c_0_n, transform=trans, 
        ha="right", va="center", fontsize=8)
trans = transforms.blended_transform_factory(
    ax1.get_yticklabels()[0].get_transform(), ax1.transData)
ax1.text(0,mean_0, "{:.3f}".format(mean_0/1e5), color=c_0, transform=trans, 
        ha="right", va="center", fontsize=8)
ax1.text(0,mean_1+1000, "{:.3f}".format(mean_1/1e5), color=c_1, transform=trans, 
        ha="right", va="center", fontsize=8)
ax1.text(0,mean_4-1000, "{:.3f}".format(mean_4/1e5), color=c_4, transform=trans, 
        ha="right", va="center", fontsize=8)

# ax1.scatter(np.arange(data_0_n.size, dtype='int'), data_0_n, color=c_0_n, label='NLP', s=25)
# ax1.scatter(np.arange(data_0.size, dtype='int'), data_0, color=c_0, label='S0R', s=12)

ax1.scatter(np.arange(data_0_n.size, dtype='int')+1, data_0_n, facecolors='none', linewidth=0.8, edgecolors=c_0_n, label='NLP', s=20)
ax1.scatter(np.arange(data_0.size, dtype='int')+1, data_0, marker='+', linewidth=0.8, color=c_0, label='2S-0R', s=20)
# ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, color=c_1, label='2S-1R', s=7)
ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, linewidth=0.8, marker='x', color=c_1, label='2S-1R', s=20)
ax1.scatter(np.arange(data_4.size, dtype='int')+1, data_4, color=c_4, label='2S-5R',s=2.5)
# # ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, marker='^', color=c_1, label='S1R', s=20)
# ax1.scatter(np.arange(data_4.size, dtype='int')+1, data_4, linewidth=0.7, marker='.', color=c_4, label='S5R', s=12)

# ax1.scatter(np.arange(data_0_n.size, dtype='int'), data_0_n, facecolors='none', edgecolors=c_0_n, label='S1R', s=25)
# ax1.scatter(np.arange(data_0.size, dtype='int')+1, data_0, marker='v', linewidth=0.7, color=c_0, label='S0R', s=15)
# ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, linewidth=0.7, marker='^', color=c_1, label='NLP', s=15)
# # ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, marker='^', color=c_1, label='S1R', s=20)
# ax1.scatter(np.arange(data_4.size, dtype='int')+1, data_4, linewidth=0.7, marker='.', color=c_4, label='S5R', s=12)


# ax1.scatter(np.arange(data_0_n.size, dtype='int')+1, data_0_n, facecolors='none', linewidth=0.5, edgecolors=c_0_n, label='NLP', s=35)
# ax1.scatter(np.arange(data_0.size, dtype='int')+1, data_0, facecolors='none', linewidth=0.5, edgecolors=c_0, label='S0R', s=20)
# ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, facecolors='none', linewidth=0.5, edgecolors=c_1, label='S1R', s=10)
# ax1.scatter(np.arange(data_4.size, dtype='int')+1, data_4, color=c_4, label='S5R',s=2)

# ax1.set_title('Convergence plot', fontsize=16)
ax1.set_xlabel('Random starting point N', fontsize=10) #Equivalent to footnotesize if pt = 10 https://tex.stackexchange.com/questions/24599/what-point-pt-font-size-are-large-etc
ax1.set_ylabel('Volume', fontsize=10)
ax1.set_xticks([1,25,50])
ax1.legend(fontsize=8)
ax1.set_axisbelow(True)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
plt.tight_layout()

# Histogram plot for the distribution
# ax2.set_title('Optimized design distribuition', fontsize=16)
# ax2.set_xlabel('Frequency', fontsize=12)
# ax2.hist([data_0,data_1,data_4], bins=18, color=[c_0,c_1,c_4], range=[80000,150000], orientation="horizontal")
#ax2.hist(data_0, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
#ax2.hist(data_1, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
#ax2.hist(data_4, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)

name = '10bar-conv'
# tikzplotlib.save(name+'.tex')
plt.savefig(name+'.pdf')
plt.savefig(name+'.png', dpi=300)
plt.savefig(name+'.pgf')