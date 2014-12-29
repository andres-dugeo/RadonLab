# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import random
import Radon
import math
sp_short=[-5,-4,-3,3,4,5]
sp=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
rd=np.linspace(- 1, 1, len(sp))
n=len(sp)
m=60

seed = random.randint(0, 2**16)
seed = 14723
print "seed:", seed
random.seed(seed)

radon = Radon.Radon(sp, rd, m, 1)
radon_pruned = Radon.Radon(sp_short, rd, m, 1)

#input = np.array([[random.uniform(-1,1)*(i[==i[1]) for i in zip([random.randint(5,5)]*n,range(n))] for i in range(m)]).transpose()
input = np.array([[math.sin(3.14 /2 *j)*(i[0]==i[1]) for i in zip([random.randint(4,6)]*n,range(n))] for j in range(m)]).transpose()

output = radon_pruned.adjointInverse(input)

s = radon_pruned.newScheme(output, 45)

input_byInv = radon_pruned.adjointForward(output)
output_byInv = radon.adjointInverse(input_byInv)

output_full = radon.adjointInverse(input)
output_full_new = radon.adjointInverse(s.matrix_1)

f,(pl1, pl2) = plot.subplots(2,3)
pl1[0].imshow(input.transpose(), interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=1, vmin=-1, aspect='auto')
pl2[0].imshow(output_full.transpose(), cmap=cm.get_cmap("seismic"), vmax=2, vmin=-2, aspect='auto')
pl1[2].imshow(s.matrix_1.transpose(), interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=1, vmin=-1, aspect='auto')
pl2[2].imshow(output_full_new.transpose(), cmap=cm.get_cmap("seismic"), vmax=2, vmin=-2, aspect='auto')
pl1[1].imshow(input_byInv.transpose(), interpolation="nearest", cmap=cm.get_cmap("seismic"), vmax=1, vmin=-1, aspect='auto')
pl2[1].imshow(output_byInv.transpose(), cmap=cm.get_cmap("seismic"), vmax=2, vmin=-2, aspect='auto')

pl1[0].set_title("Input")
pl1[1].set_title("Radon")
pl1[2].set_title("Modified Radon")

plot.show()
