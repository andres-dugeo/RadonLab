# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import random
import Radon_weighted as Radon
import Radon as Radon_nw
import math

sp_short = [-5,-4,-3,3,4,5]
sp = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
weight = [1,1,1,.5,.5,.5,.5,.5,1,1,1]
rd = np.linspace(- 1, 1, len(sp))
n = len(sp)
n_pruned = len(sp_short)

output = np.load('./data.npy')[0:11,400:450]

print output.shape

m = output.shape[1]

seed = random.randint(0, 2**16)
print "seed:", seed
random.seed(seed)

radon = Radon_nw.Radon(sp, rd, m, 1)
radon_weighted = Radon.Radon(sp, weight, rd, m, 1)
radon_pruned = Radon_nw.Radon(sp_short, rd, m , 1)

output_pruned = np.zeros((n_pruned,m))
for i in range(n_pruned):
    index = sp.index(sp_short[i])
    for j in range(m):
        output_pruned[(i,j)] = output[(index,j)]

s_pruned = radon_pruned.newScheme(output_pruned, 200)


s = radon_weighted.newScheme(output, 200)
input = radon.adjointForward(output)

output_full = output
output_full_new = radon_weighted.adjointInverse(s.matrix_1)

output_diff = radon.inverse(s.matrix_3)

f,(pl1, pl2) = plot.subplots(2,4)
pl1[0].imshow(input.transpose(),interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=10000000, vmin=-10000000, aspect='auto')
pl2[0].imshow(output_full.transpose(), cmap=cm.get_cmap("seismic"), vmax=100000, vmin=-100000, aspect='auto')
pl1[1].imshow(s_pruned.matrix_1.transpose(), interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=1000000, vmin=-1000000, aspect='auto')
pl2[1].imshow(radon.adjointInverse(s_pruned.matrix_1).transpose(), cmap=cm.get_cmap("seismic"), vmax=100000, vmin=-100000, aspect='auto')
pl1[2].imshow(s.matrix_1.transpose(), interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=10000000, vmin=-10000000, aspect='auto')
pl2[2].imshow(output_full_new.transpose(), cmap=cm.get_cmap("seismic"), vmax=100000, vmin=-100000, aspect='auto')
pl1[3].imshow((s.matrix_1+radon.adjointForward(output_diff)).transpose(), interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=100000, vmin=-100000, aspect='auto')
pl2[3].imshow((output_full_new+output_diff).transpose(), cmap=cm.get_cmap("seismic"), vmax=100000, vmin=-100000, aspect='auto')

pl1[0].set_title("Input")
pl1[1].set_title("Modified Radon")
pl1[2].set_title("weighted MR")
pl1[3].set_title("wMR enhanced")

input_tp = radon.forward(output)
plot.figure()
plot.imshow(input_tp.transpose(),  interpolation="nearest" , cmap=cm.get_cmap("seismic"), vmax=1000000, vmin=-1000000, aspect='auto')

plot.show()
