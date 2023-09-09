import matplotlib
#import matplotlib.pyplot as plt;

sample = 50

#normalize item number values to colormap
norm = matplotlib.colors.Normalize(vmin=0, vmax=sample-1)

cmap = matplotlib.cm.get_cmap('coolwarm')

a = cmap(norm(0),bytes=True)

with open('coolwarm.txt', 'w') as f:
    for i in range(sample):
        f.write('rgb255({0}cm)=({1},{2},{3});\n'.format(i,*cmap(norm(i),bytes=True)[:3]))