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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# to change default colormap
plt.rcParams["image.cmap"] = "coolwarm"
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

c_nlp = cmap(norm(7))
c_0 = cmap(norm(1.5))
c_4 = cmap(norm(0))

data_0 = np.load('0.npy')
data_nlp = np.load('nlp.npy')

data_4 = np.load('4.npy')[0:100]

nn = np.count_nonzero(data_nlp>data_4)


mm = 1/25.4  # mm in inches
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160*mm,80*mm), sharey=True, gridspec_kw={'width_ratios': [2.2, 1]})
ax1.grid(True)
# Calculate statistical measure of interest
mean_0 = np.mean(data_0)
std_dev_0 = np.std(data_0)
mean_nlp = np.mean(data_nlp)
std_dev_nlp = np.std(data_nlp)
mean_4 = np.mean(data_4)
std_dev_4 = np.std(data_4)

achtz_opt = 85.575
# Plot optimum value without chain buckling
nochain_opt = 99.988
# ax1.axhline(y = nochain_opt, color = 'black', linestyle = 'dashed', label = "Optimum w/o chain buck.")

# Plot mean
ax1.axhline(y = mean_nlp, color = c_nlp,  linewidth=1, linestyle = 'dashed', zorder = 0.5)
ax1.axhline(y = mean_0, color = c_0, linewidth=1, linestyle = 'dashed', zorder = 0.5)
ax1.axhline(y = mean_4, color = c_4, linewidth=1, linestyle = 'dashed', zorder = 0.5)

trans = transforms.blended_transform_factory(
    ax1.get_yticklabels()[0].get_transform(), ax1.transData)
ax1.text(0,achtz_opt-0.5, "{:.2f}".format(achtz_opt), color='grey', transform=trans, 
        ha="right", va="center", fontsize=8)
ax1.text(0, mean_nlp +0.7, "{:.2f}".format(mean_nlp), color=c_nlp, transform=trans, 
        ha="right", va="center", fontsize=8)
ax1.text(0,mean_0, "{:.2f}".format(mean_0), color=c_0, transform=trans, 
        ha="right", va="center", fontsize=8)
ax1.text(0,mean_4+0.5, "{:.2f}".format(mean_4), color=c_4, transform=trans, 
        ha="right", va="center", fontsize=8)

ax1.scatter(np.arange(data_nlp.size, dtype='int')+1, data_nlp, facecolors='none', edgecolors=c_nlp, label='NLP', s=20)
ax1.scatter(np.arange(data_0.size, dtype='int')+1, data_0, linewidth=1.2, marker='+', color=c_0, label='2S-0R', s=35)
ax1.scatter(np.arange(data_4.size, dtype='int')+1, data_4, color=c_4, label='2S-5R',s=4)

# Plot achtzinger optimum value
ax1.axhline(y = achtz_opt, linewidth=1, color = 'grey', linestyle = 'dashed', label = "Achtzinger (1999b)", zorder = 1)

# ax1.set_title('Convergence plot', fontsize=16)
ax1.set_xlabel('Random starting point N', fontsize=8) #Equivalent to footnotesize if pt = 11 https://tex.stackexchange.com/questions/24599/what-point-pt-font-size-are-large-etc
ax1.set_ylabel('Volume', fontsize=8)
ax1.set_xticks([1,20,40,60,80,100])

ax1.legend(fontsize=8)
ax1.set_axisbelow(True)
ax1.tick_params(axis='both', which='major', labelsize=8)

# Histogram plot for the distribution
# ax2.set_title('Optimized design distribuition', fontsize=16)
ax2.set_xlabel('Frequency', fontsize=8)
label = [r'$\bar V=$ {0:.2f}, SD $=$ {1:.2f}'.format(mean_nlp,std_dev_nlp),r'$\bar V=$ {0:.2f}, SD $=$ {1:.2f}'.format(mean_0,std_dev_0), r'$\bar V=$ {0:.2f}, SD $=$ {1:.2f}'.format(mean_4,std_dev_4)]
ax2.hist([data_nlp,data_0,data_4], bins=10, alpha=1, range=[75,125], color=[c_nlp,c_0,c_4], orientation="horizontal", label=label)
#ax2.hist(data_0, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
# ax2.hist(data_nlp, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
#ax2.hist(data_4, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
ax2.legend(fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))


plt.tight_layout()

name = 'cant_ach-conv'
# tikzplotlib.save(name+'.tex')
plt.savefig(name+'.pdf')
plt.savefig(name+'.png', dpi=300)
plt.savefig(name+'.pgf')