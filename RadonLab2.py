import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import random
import Radon
import math
sp_short=[-6,-5,-4,4,5,6]
sp=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
rd=np.linspace(- 1, 1, len(sp))
n = len(sp)
n_pruned = len(sp_short)
m = 40

seed = random.randint(0, 2**16)
print "seed:", seed
random.seed(seed)

radon = Radon.Radon(sp, rd, m, 1)
radon_pruned = Radon.Radon(sp_short, rd, m, 1)

output = np.zeros((n,m))
for i in range(n/2+1):
	for k in range(15):
		j = 5.0 + i / 2.0 + 2.0*k 
		lower_index = math.floor(j)
		upper_index = 1 + lower_index
		lower_fraction = ((-1)**k)*(1 - j + lower_index)
		upper_fraction = ((-1)**k)*(j - lower_index)
		output[(i, lower_index)] = lower_fraction
		output[(i, upper_index)] = upper_fraction
		output[(n-i-1, lower_index)] = lower_fraction
		output[(n-i-1, upper_index)] = upper_fraction
	#print j, lower_index, upper_index, lower_fraction, upper_fraction

output_pruned = np.zeros((n_pruned,m))
for i in range(n_pruned):
	index = sp.index(sp_short[i])
	for j in range(m):
		output_pruned[(i,j)] = output[(index,j)]


s = radon_pruned.newScheme(output_pruned, 150)

input_byInv = radon_pruned.adjointForward(output_pruned)
output_byInv = radon.adjointInverse(input_byInv)

input = radon.adjointForward(output)

output_full = output
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
