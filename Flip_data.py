#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 00:38:17 2026

@author: sergi
"""

import os
from itertools import permutations, combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from natsort import natsorted
import sklearn as sk
import time
from joblib import Parallel, delayed
import shutil


def calcul_distancia(i, j, solucions, permutacions):
    distancia = np.inf
            
    for p in permutacions:
        diccionari = {k: p[k-1] for k in range(1, 8)} #Dictionary to relabel IDs using a given permutation
            
        permutat = solucions[j].copy()
        permutat['TERRITORY_ID'] = permutat['TERRITORY_ID'].map(diccionari) #Applies permutation
        
        nova_distancia = (solucions[i]['TERRITORY_ID'] != permutat['TERRITORY_ID']).sum() #Hamming distance
        
        if nova_distancia < distancia:
            
            distancia = nova_distancia #Stores best (minimum) distance 
    return i, j, distancia

def load_minimum_repartiments(minimum_energy, directory = 'Solutions'):
    minimum_dfs = []
    os.makedirs('Optimal_maps', exist_ok=True)
    seed = minimum_energy['Seed'].values
    flips = minimum_energy['Flips'].values
    for s, f in zip(seed, flips):
        carpeta = os.path.join(directory, f'{f}flips')
        fitxer = f'Repartiment_seed{int(s)}_iter10000_{int(f)}flips.pkl'
        cami = os.path.join(carpeta, fitxer)
        batch = pd.read_pickle(cami)
        minimum_dfs.append(batch)
        
        mapa_origen = os.path.join('Maps', f'{f}flips', f'map_seed{s}_iter10000_{f}flips.png')
        mapa_desti = os.path.join('Optimal_maps', f'map_seed{s}_iter10000_{f}flips.png')
        if os.path.exists(mapa_origen):
            shutil.copy2(mapa_origen, mapa_desti)
    return minimum_dfs

directory = 'Solutions'

#Statistics of solution per value of flips/iteration

flip_data = pd.DataFrame()
for entry in os.scandir(directory):
    if entry.is_dir():
        folder = entry.name
        batch = pd.DataFrame()
        
        repartiments = [file for file in os.listdir(os.path.join(directory, folder)) if re.match('^Repartiment', file)] #Find filenames
        totals = [file for file in os.listdir(os.path.join(directory, folder)) if re.match('^Totals', file)]
        repartiments = natsorted(repartiments) #It automatically sorts by seed due to naming convention of the files
        
        solucions = [pd.read_pickle(f'{directory}/{folder}/{sol}') for sol in repartiments] #Load dataframes
        agregats = [pd.read_pickle(f'{directory}/{folder}/{tot}') for tot in totals] #Load aggregate dataframes
        batch['Energy'] = [df['Desviacio'].sum() for df in agregats]
        numbers = [re.findall(r'\d+', file) for file in os.listdir(os.path.join(directory, folder)) if re.match('^Totals', file)]
        batch['Seed'] = [int(number[0]) for number in numbers]
        batch['Flips'] = [int(number[2]) for number in numbers]
        
        flip_data = pd.concat([flip_data, batch], ignore_index=True)
        
flip_agregat = flip_data.groupby('Flips', as_index = False).agg(Energy=('Energy' , 'mean'), Sigma=('Energy', 'sem'))

flip_agregat['Signficancia'] = (flip_agregat['Energy']-flip_agregat['Energy'][2])/np.sqrt(flip_agregat['Sigma']**2 + flip_agregat['Sigma'][2]**2)

flip_data.to_pickle('Flip_data.pkl')

#Minimum energy solutions Hamming distance
permutacions = list(permutations(range(1,8)))
minimum_energy = flip_data[flip_data['Energy'] == flip_data['Energy'].min()]

minimum_dfs = load_minimum_repartiments(minimum_energy)
parells = list(combinations(range(len(minimum_dfs)), 2))

resultats = Parallel(n_jobs=12)(delayed(calcul_distancia)(i, j, minimum_dfs, permutacions) for i, j in parells)
distance_minimums = np.zeros((len(minimum_dfs), len(minimum_dfs)))
for i, j, distancia in resultats:              
    distance_minimums[i, j] = distancia #Result is transfered to distance matrix
    distance_minimums[j, i] = distance_minimums[i, j] #Distance is symmetric
    
mask_diagonal = np.triu_indices_from(distance_minimums, 1) #Mask for upper triangle
upper_diagonal = distance_minimums[mask_diagonal]
count, bins = np.histogram(upper_diagonal, np.arange(np.min(upper_diagonal), np.max(upper_diagonal)+1, 1))

plt.figure()
plt.bar(bins[:-1], count)
plt.xlabel('Hamming distance')
plt.ylabel('Count')
filename = 'Mimimum_Hamming.png'
plt.savefig(filename, dpi = 300)
plt.close()

# Plotting histograms and mean energy vs flips

fig, ax = plt.subplots()

ax.errorbar(flip_agregat['Flips'], flip_agregat['Energy'], flip_agregat['Sigma'], fmt='o', ms=4, linewidth=2, capsize=6)
ax.set_xlabel('Flips per iteration')
ax.set_ylabel('Mean energy (inhabitants)')
plt.tight_layout()
plt.savefig('Energy_vs_flips.png', dpi = 300)
plt.close()


dist_folder = 'Distances'
histo_data = pd.DataFrame(columns = ['Flips', 'Mean', 'Sigma', 'Q1', 'Q3'])
for entry in os.scandir(dist_folder):
    if entry.is_file():
        file = entry.name
        hamming = np.load(f'{dist_folder}/{file}')
        
        mask_diagonal = np.triu_indices_from(hamming, 1) #Mask for upper triangle
        upper_diagonal = hamming[mask_diagonal]
        count, bins = np.histogram(upper_diagonal, np.arange(np.min(upper_diagonal), np.max(upper_diagonal)+1, 1))
        nombres = re.findall(r'\d+', file)
        
        q1, q3 = np.percentile(upper_diagonal, [25, 75])
        histo_data.loc[-1] = [int(nombres[0]), np.mean(upper_diagonal), np.std(upper_diagonal), q1, q3]
        
        histo_data.index += 1
        
        
fig2, ax2 = plt.subplots()

ax2.errorbar(histo_data['Flips'], histo_data['Mean'], histo_data['Sigma'], fmt='o', ms=4, linewidth=2, capsize=6)
ax2.set_xlabel('Flips per iteration')
ax2.set_ylabel('Mean Hamming distance')
plt.tight_layout()
plt.savefig('AvgHamming_vs_flips.png', dpi = 300)
plt.close()