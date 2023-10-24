import numpy as np

a = np.load("data/a.npy")
GS = np.load("data/GS.npy")
nodes = np.load("data/nodes.npy")
q = np.load("data/q.npy")

l = np.linalg.norm((nodes[GS[:,0],1]-nodes[GS[:,1],1], nodes[GS[:,0],0]-nodes[GS[:,1],0]),axis=0)

v = np.sum(a*l)

with open('tab'+'.txt', 'w') as f:
    for i in range(a.size):
        f.write('({0:d} {1:d})\t({2:d} {3:d})\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\n'.format(int(nodes[GS[i,0],0]),
                                                                                          int(nodes[GS[i,0],1]),
                                                                                          int(nodes[GS[i,1],0]),
                                                                                          int(nodes[GS[i,1],1]),
                                                                                          l[i],
                                                                                          q[i,0],
                                                                                          q[i,1],
                                                                                          a[i],
                                                                                          a[i]*l[i]
                                                                                          )) 
    