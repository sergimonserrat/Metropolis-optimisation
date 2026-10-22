#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:29:03 2026

@author: sergi
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from itertools import permutations, combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from natsort import natsorted
import sklearn as sk
import time
from joblib import Parallel, delayed
from numba import njit

start = time.time()
'''
This script computes the minimum Hamming distance between two solutions. It takes into account the different permutations
due to different labeling of the regions. That is, 2 solutions could be the same but score 51 in Hamming distance due to
a different TERRITORY_ID labeling.

100 solutions x 4 values of flips ~6 hours
'''
histo_folder = 'Histograms'
os.makedirs(histo_folder, exist_ok = True)
print(f'Histograms will be stored in {histo_folder} folder')

dist_folder = 'Distances'
os.makedirs(dist_folder, exist_ok=True)
print(f'Distance matrices will be stored in {dist_folder} folder')

permutacions = np.array(list(permutations(range(1,8)))) #Generate permutations
directory = 'Solutions'

@njit
def calcul_distancia(i, j, solucions, permutacions):
    distancia = np.inf
            
    for p in permutacions:
        nova_distancia = np.sum(solucions[i] != p[solucions[j] - 1])
        if nova_distancia < distancia:
            distancia = nova_distancia #Stores best (minimum) distance
    return i, j, distancia

for entry in os.scandir(directory):
    if entry.is_dir() and entry.name:
        folder = entry.name

        print(f'Processing {folder}...')
    
        repartiments = [file for file in os.listdir(os.path.join(directory, folder)) if re.match('^Repartiment', file)] #Find filenames
        totals = [file for file in os.listdir(os.path.join(directory, folder)) if re.match('^Totals', file)]
        n = len(repartiments) #Number of solutions
        repartiments = natsorted(repartiments) #It automatically sorts by seed due to naming convention of the files
        hamming = np.zeros((n, n)) #Initialise distance matrix
        
        solucions = [pd.read_pickle(f'{directory}/{folder}/{sol}') for sol in repartiments] #Load dataframes
        solucions = np.array([np.array(sol['TERRITORY_ID']) for sol in solucions])
        agregats = [pd.read_pickle(f'{directory}/{folder}/{tot}') for tot in totals] #Load aggregate dataframes
        energies = [df['Desviacio'].sum() for df in agregats]
        #energies = energies[:10]
        
        parells = list(combinations(range(n), 2))
        resultats = Parallel(n_jobs=12)(delayed(calcul_distancia)(i, j, solucions, permutacions) for i, j in parells)
        
        for i, j, distancia in resultats:              
            hamming[i, j] = distancia #Result is transfered to distance matrix
            hamming[j, i] = hamming[i, j] #Distance is symmetric
        
        
        MDS = sk.manifold.MDS(n_components = 3, metric = 'precomputed', init = 'random', max_iter = 3000, metric_mds = False, n_init=20)
        np.save(f'{dist_folder}/Hamming_{folder}.npy', hamming)
        '''
        metric_mds = True yielded stresses too high so it is set to False.
        '''
        xy = MDS.fit_transform(hamming)
        stress = MDS.stress_
        print(f'Stress is: {stress:.2f}')
        
        '''
        The solutions don't cluster and the histogram shows most solutions are separated by 15-25 flips so I guess this proves 
        the many local minima landscape I hypothesised. A move to quantum optimisation seems justified.
        '''
        #Plot MDS
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        ax.set_zlabel('MDS Dimension 3')
        plot = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c = energies)
        colorbar = fig.colorbar(plot, ax=ax)
        colorbar.set_label('Sum of deviations from the mean (energy)')
        plt.tight_layout()
        plt.savefig(f'Solution_landscape_{folder}.png', dpi = 300)
        plt.close()
        
        mask_diagonal = np.triu_indices_from(hamming, 1) #Mask for upper triangle
        upper_diagonal = hamming[mask_diagonal]
        count, bins = np.histogram(upper_diagonal, np.arange(np.min(upper_diagonal), np.max(upper_diagonal)+1, 1))
        
        #Plot histogram
        plt.figure()
        plt.bar(bins[:-1], count)
        plt.xlabel('Hamming distance')
        plt.ylabel('Count')
        filename = f'{histo_folder}/Hamming_{folder}.png'
        plt.savefig(filename, dpi = 300)
        plt.close()

end = time.time()

print('Time elapsed:', end-start)