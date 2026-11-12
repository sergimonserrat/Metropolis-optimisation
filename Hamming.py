#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:29:03 2026

@author: sergi
"""

import argparse
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
#import sklearn as sk
import time
from joblib import Parallel, delayed
from numba import njit
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph',
        choices=['synthetic', 'real'],
        default='synthetic',
        help='Type of graph that was partitioned (default: synthetic).',
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        default=None,
        help='List of folders to analyse (e.g. --folders Synth0 Synth3). '
             'If omitted, all folders found are analysed.',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=12,
        help='Number of parallel jobs (default: 12).',        
    )
    return parser.parse_args()


args = arg_parse()

start = time.time()

directory = f'{ROOT}/SynthSolutions' if args.graph == 'synthetic' else f'{ROOT}/Solutions'
histo_folder = f'{ROOT}/Histograms'
os.makedirs(histo_folder, exist_ok = True)
print(f'Histograms will be stored in {histo_folder} folder')

dist_folder = f'{ROOT}/Distances'
os.makedirs(dist_folder, exist_ok=True)
print(f'Distance matrices will be stored in {dist_folder} folder')
    
permutacions = np.array(list(permutations(range(1,8)))) #Generate permutations

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
        graph_folder = entry.name
        
        os.makedirs(f'{dist_folder}/{graph_folder}', exist_ok=True)
        if args.folders is not None and graph_folder not in args.folders:
            continue
        
        for subentry in os.scandir(entry.path):
            if subentry.is_dir() and subentry.name:
                folder = subentry.name
                
                print(f'Processing {folder}...')
            
                repartiments = [file for file in os.listdir(subentry.path) if re.match('^Repartiment', file)] #Find filenames
                #totals = [file for file in os.listdir(subentry.path) if re.match('^Totals', file)]
                n = len(repartiments) #Number of solutions
                repartiments = natsorted(repartiments) #It automatically sorts by seed due to naming convention of the files
                hamming = np.zeros((n, n)) #Initialise distance matrix
                
                solucions = [pd.read_pickle(f'{subentry.path}/{sol}') for sol in repartiments] #Load dataframes
                solucions = np.array([np.array(sol['TERRITORY_ID']) for sol in solucions])
                #agregats = [pd.read_pickle(f'{subentry.path}/{tot}') for tot in totals] #Load aggregate dataframes
                #energies = [df['Desviacio'].sum() for df in agregats]
                #energies = energies[:10]
                
                parells = list(combinations(range(n), 2))
                resultats = Parallel(n_jobs=args.jobs)(delayed(calcul_distancia)(i, j, solucions, permutacions) for i, j in parells)
                
                for i, j, distancia in resultats:              
                    hamming[i, j] = distancia #Result is transfered to distance matrix
                    hamming[j, i] = hamming[i, j] #Distance is symmetric
                
                np.save(f'{dist_folder}/{graph_folder}/Hamming_{folder}.npy', hamming)
                '''
                MDS = sk.manifold.MDS(n_components = 3, metric = 'precomputed', init = 'random', max_iter = 3000, metric_mds = False, n_init=20)
                #metric_mds = True yielded stresses too high so it is set to False.
                
                xy = MDS.fit_transform(hamming)
                stress = MDS.stress_
                print(f'Stress is: {stress:.2f}')
                

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
                plt.savefig(f'{ROOT}/Solution_landscape_{folder}.png', dpi = 300)
                plt.close()
                '''
                mask_diagonal = np.triu_indices_from(hamming, 1) #Mask for upper triangle
                upper_diagonal = hamming[mask_diagonal]
                count, bins = np.histogram(upper_diagonal, np.arange(np.min(upper_diagonal), np.max(upper_diagonal)+1, 1))
                
                os.makedirs(f'{histo_folder}/{graph_folder}', exist_ok=True)
                #Plot histogram
                plt.figure()
                plt.bar(bins[:-1], count)
                plt.xlabel('Hamming distance')
                plt.ylabel('Count')
                filename = f'{histo_folder}/{graph_folder}/Hamming_{folder}.png'
                plt.savefig(filename, dpi = 300)
                plt.close()

end = time.time()

print('Time elapsed:', end-start)