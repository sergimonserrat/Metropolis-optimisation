#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:29:03 2026

@author: sergi
"""

from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from natsort import natsorted
import sklearn as sk
'''
This script computes the minimum Hamming distance between two solutions. It takes into account the different permutations
due to different labeling of the regions. That is, 2 solutions could be the same but score 51 in Hamming distance due to
a different TERRITORY_ID labeling.

Tried it for 100 solutions. Takes several hours to run. With 10 solutions it takes just a few minutes.
'''

permutacions = list(permutations(range(1,8))) #Generate permutations
directory = 'Solutions'
repartiments = [file for file in os.listdir(directory) if re.match('^Repartiment', file)] #Find filenames
totals = [file for file in os.listdir(directory) if re.match('^Totals', file)]
n = 10#len(repartiments) #Number of solutions
repartiments = natsorted(repartiments) #It automatically sorts by seed due to naming convention of the files
hamming = np.full((n,n),np.inf) #Initialise distance matrix

solucions = [pd.read_pickle(f'{directory}/{sol}') for sol in repartiments] #Load dataframes
agregats = [pd.read_pickle(f'{directory}/{tot}') for tot in totals] #Load aggregate dataframes
energies = [df['Desviacio'].sum() for df in agregats]
energies = energies[:10]
for i in range(n):
    hamming[i, i] = 0 #Solutions are at zero distance to themselves
    for j in range(i+1, n):
        print('Computing distance between seed', i, 'and seed', j)
        sol_i = solucions[i]
        sol_j = solucions[j]
        distancia = np.inf
        
        for p in permutacions:
            diccionari = {k: p[k-1] for k in range(1, 8)} #Dictionary to relabel IDs using a given permutation
            
            permutat = sol_j.copy()
            permutat['TERRITORY_ID'] = permutat['TERRITORY_ID'].map(diccionari) #Applies permutation
            
            nova_distancia = (sol_i['TERRITORY_ID'] != permutat['TERRITORY_ID']).sum() #Hamming distance
            
            if nova_distancia < distancia:
                
                distancia = nova_distancia #Stores best (minimum) distance
                print(distancia)
        
        
        hamming[i, j] = distancia #Result is transfered to distance matrix
        hamming[j, i] = hamming[i, j] #Distance is symmetric


MDS = sk.manifold.MDS(n_components = 3, metric = 'precomputed', init = 'random', max_iter = 3000, metric_mds = False, n_init=20)
'''
metric_mds = True yielded stresses too high so it is set to False.
'''
xy = MDS.fit_transform(hamming)
stress = MDS.stress_
print(f'Stress is: {stress:.2f}')

''''
Just discovered this MDS thing about a week ago. I am not an expert, but the solutions don't cluster and the histogram
shows most solutions are separated by 15-25 flips so I guess this proves the many local minima landscape I hypothesised.
A move to quantum optimisation looks justified.
'''
#Plot MDS
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.set_title('Sample of the energy landscape')
ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')
ax.set_zlabel('MDS Dimension 3')
plot = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c = energies)
colorbar = fig.colorbar(plot, ax=ax)
colorbar.set_label('Sum of deviations from the mean (energy)')
plt.tight_layout()
plt.savefig('Solution_landscape.png', dpi = 300)
plt.close()

mask_diagonal = np.triu_indices_from(hamming, 1) #Mask for upper triangle
upper_diagonal = hamming[mask_diagonal]
count, bins = np.histogram(upper_diagonal, np.arange(np.min(upper_diagonal), np.max(upper_diagonal)+1, 1))

#Plot histogram
plt.figure()
plt.bar(bins[:-1], count)
plt.xlabel('Hamming distance')
plt.ylabel('Count')
plt.title('Histogram of Hamming distances for 100 solutions')
filename = 'Hamming.png'
plt.savefig(filename, dpi = 300)
plt.close()