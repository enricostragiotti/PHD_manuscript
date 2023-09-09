import matplotlib

sample = 8
norm = matplotlib.colors.Normalize(vmin=0, vmax=sample-1) #normalize item number values to colormap
cmap = matplotlib.cm.get_cmap('coolwarm')