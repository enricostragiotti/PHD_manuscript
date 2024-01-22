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

c_1= cmap(norm(0))
c_2 = cmap(norm(7))

# Stiffness definition
E = 69e3 # [MN/m2] MPa - alu
# Stress_c vaulues definition
s_c = -270 # [MN/m2] MPa - alu
s_t = 270 # [MN/m2] MPa - alu
# Density
rho = 2.7 * 1e3 # [kg/m3] - alu

# Section of the members
s_buck = np.pi * E / 4 # Circular sections

# Safety factor
sf = np.array([1.5, 1.5, 2.67])

a_lp = np.load('a_LP.npy')
a = np.load('a.npy')
q_lp = np.load('q_LP.npy')
q = np.load('q.npy')
l_lp = np.loadtxt('l_LP.dat')
l = np.loadtxt('l.dat')

n_load_cases = q.shape[-1]

c=np.array([])
c_lp=np.array([])
t=np.array([])
t_lp=np.array([])
b=np.array([])
b_lp=np.array([])

for p in range(n_load_cases):
        mm = 1/25.4  # mm in inches
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
        fig, ax1 = plt.subplots(figsize=(75*mm,60*mm))
        #fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.grid(True)

        # ax1.axhline(y = mean_0_n, color = c_0_n, xmin=-5, xmax = 5, linewidth=0.5, linestyle = 'dashed', zorder = 1)
        # ax1.axhline(y = mean_0, color = c_0, linewidth=0.5, linestyle = 'dashed', zorder = 1)        
         
        # cand_lp = np.logical_and(q_lp[:,p]<0, a_lp>1e-6) # only compression
        cand_lp = np.min(np.abs(q_lp),axis=1)>1e-6 
        # cand_lp = np.abs(a_lp)>1e-6 
        q_cand_lp = q_lp[cand_lp,p]
        
        # cand = np.logical_and(q[:,p]<0, a>1e-6) # only compression
        cand = np.min(np.abs(q),axis=1)>1e-6 
        # cand = np.abs(a)>1e-6 
        q_cand = q[cand,p]
        
        buck = -q_cand / ((s_buck/l[cand]**2) * a[cand]**2 / sf[p])
        buck_lp = -q_cand_lp / ((s_buck/l_lp[cand_lp]**2) * a_lp[cand_lp]**2 / sf[p])
        stress_c = q_cand / (s_c * a[cand] / sf[p])
        stress_c_lp = q_cand_lp / (s_c * a_lp[cand_lp] / sf[p])
        stress_t = q_cand / (s_t * a[cand] / sf[p])
        stress_t_lp = q_cand_lp / (s_t * a_lp[cand_lp] / sf[p])
        
        c=np.append(c,stress_c)
        c_lp=np.append(c_lp,stress_c_lp)
        t=np.append(t,stress_t)
        t_lp=np.append(t_lp,stress_t_lp)
        b=np.append(b,buck)
        b_lp=np.append(b_lp,buck_lp)

        ax1.scatter(stress_c_lp, buck_lp, color=c_1, label='SLP', s=10)
        ax1.scatter(stress_c, buck, color=c_2, label='NLP', s=5)
        
        # ax1.set_xlim([0, 1])
        # ax1.set_ylim([0, 1])


        # ax1.scatter(np.arange(data_0_n.size, dtype='int'), data_0_n, facecolors='none', linewidth=0.7, edgecolors=c_0_n, label='S1R', s=20)
        # ax1.scatter(np.arange(data_0.size, dtype='int')+1, data_0, marker='+', linewidth=0.7, color=c_0, label='S0R', s=20)
        # ax1.scatter(np.arange(data_1.size, dtype='int')+1, data_1, linewidth=0.7, marker='x', color=c_1, label='NLP', s=20)
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
        ax1.set_xlabel('Stress_c failure criterion', fontsize=8) #Equivalent to footnotesize if pt = 10 https://tex.stackexchange.com/questions/24599/what-point-pt-font-size-are-large-etc
        ax1.set_ylabel('Buckling failure criterion', fontsize=8)
        # ax1.set_xticks([1,25,50])
        ax1.legend(fontsize=8)
        ax1.set_axisbelow(True)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        plt.tight_layout()

        # Histogram plot for the distribution
        # ax2.set_title('Optimized design distribuition', fontsize=16)
        # ax2.set_xlabel('Frequency', fontsize=12)
        # ax2.hist([data_0,data_1,data_4], bins=18, color=[c_0,c_1,c_4], range=[80000,150000], orientation="horizontal")
        #ax2.hist(data_0, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
        #ax2.hist(data_1, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
        #ax2.hist(data_4, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)

        # name = 'failure_{:d}'.format(p)
        # # tikzplotlib.save(name+'.tex')
        # plt.savefig(name+'.pdf')
        # plt.savefig(name+'.png', dpi=300)
        # plt.savefig(name+'.pgf')
        
mm = 1/25.4  # mm in inches
# fig, axs = plt.subplots(2, 2, figsize=(75*mm,75*mm), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})
fig, ax1 = plt.subplots(figsize=(75*mm,75*mm))
#fig, ax1 = plt.subplots(figsize=(6,4))
# axs[1,0].grid(True)

# # ax1.axhline(y = mean_0_n, color = c_0_n, xmin=-5, xmax = 5, linewidth=0.5, linestyle = 'dashed', zorder = 1)
# # ax1.axhline(y = mean_0, color = c_0, linewidth=0.5, linestyle = 'dashed', zorder = 1)        

buck = np.max(b.reshape(3,-1).T, axis=1)
buck_lp = np.max(b_lp.reshape(3,-1).T, axis=1)
stress_c = np.max(c.reshape(3,-1).T, axis=1)
stress_c_lp = np.max(c_lp.reshape(3,-1).T, axis=1)
stress_t = np.max(t.reshape(3,-1).T, axis=1)
tt_lp = t_lp.reshape(3,-1).T
cc_lp = c_lp.reshape(3,-1).T
stress_t_lp = np.max(t_lp.reshape(3,-1).T, axis=1)
maxx = np.hstack([stress_c.reshape(-1,1), stress_t.reshape(-1,1)])
maxx_lp = np.hstack([stress_c_lp.reshape(-1,1), stress_t_lp.reshape(-1,1)])
stress = np.max(np.hstack([stress_c.reshape(-1,1), stress_t.reshape(-1,1)]), axis=1)
stress_lp = np.max(np.hstack([stress_c_lp.reshape(-1,1), stress_t_lp.reshape(-1,1)]), axis=1)

# axs[1,0].scatter(stress_lp, buck_lp, color=c_1, label='SLP', s=3)
# axs[1,0].scatter(stress, buck, color=c_2, label='NLP', s=1)
# line, label = axs[1,0].get_legend_handles_labels()
# axs[1,0].set_xlim([-0.02, 1.02])
# axs[1,0].set_ylim([-0.02, 1.02])


# axs[1,0].set_xlabel('Stress failure criterion', fontsize=8) #Equivalent to footnotesize if pt = 10 https://tex.stackexchange.com/questions/24599/what-point-pt-font-size-are-large-etc
# axs[1,0].set_ylabel('Buckling failure criterion', fontsize=8)
# # axs[1,0].set_xticks([1,25,50])
# axs[1,0].tick_params(axis='both', which='major', labelsize=8)
# # axs[1,0].legend(fontsize=8)
# axs[1,0].set_axisbelow(True)

# # l4 = axs[1,0].legend(bbox_to_anchor=(0, -0.5, 1, 0.2), loc="lower left",
#                 # mode="expand", borderaxespad=0, ncol=2)

# # Shrink current axis's height by 10% on the bottom
# # box = axs[1,0].get_position()
# # axs[1,0].set_position([box.x0, box.y0 + box.height * 0.1,
# #                  box.width, box.height * 0.9])

# # # Put a legend below current axis
# # axs[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=8, ncol=2)

# axs[0,1].axis('off')
# axs[0,1].legend(line, label, fontsize = 8)
# axs[0,0].hist([stress,stress_lp], bins=5, alpha=1, range=[-0.02,1.02], color=[c_2,c_1])
# axs[1,1].hist([buck,buck_lp], bins=5, alpha=1, range=[-0.02,1.02], color=[c_2,c_1], orientation="horizontal")

# axs[0,0].tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False, # labels along the bottom edge are off
#     labelsize=8) 

# axs[1,1].tick_params(
#     axis='y',          # changes apply to the y-axis
#     which='both',      # both major and minor ticks are affected
#     left=False,      
#     top=False,        
#     labelleft=False,
#     labelsize = 8) 

# axs[0,0].set_xlim([-0.02, 1.02])
# axs[1,1].set_ylim([-0.02, 1.02])

# ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

ID = np.logical_and(stress<0.95, buck<0.95) 
IDD = np.where(ID)[0]


ax1.scatter(stress_lp, buck_lp, color=c_1, label='SLP', linewidth=0.8, marker='+', s=25)
ax1.scatter(stress, buck, color=c_2, label='NLP', linewidth=0.8, marker='x', s=10)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('Stress failure criterion ($c_s$)', fontsize=8) #Equivalent to footnotesize if pt = 10 https://tex.stackexchange.com/questions/24599/what-point-pt-font-size-are-large-etc
ax1.set_ylabel('Buckling failure criterion ($c_b$)', fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.legend(fontsize=8)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.legend(bbox_to_anchor=(0.5, 1.1, 0,0), loc="center", ncol=2, fontsize=8)


plt.tight_layout()

# Histogram plot for the distribution
# ax2.set_title('Optimized design distribuition', fontsize=16)
# ax2.set_xlabel('Frequency', fontsize=12)
# ax2.hist([data_0,data_1,data_4], bins=18, color=[c_0,c_1,c_4], range=[80000,150000], orientation="horizontal")
#ax2.hist(data_0, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
#ax2.hist(data_1, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)
#ax2.hist(data_4, bins=18, alpha=1, range=[80000,150000], orientation="horizontal", histtype='step', fill=False)

name = 'failure_max'
plt.savefig(name+'.pdf')
plt.savefig(name+'.png', dpi=300)