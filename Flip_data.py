#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 00:38:17 2026

@author: sergi
"""

import argparse
import os
from itertools import permutations, combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from natsort import natsorted
from joblib import Parallel, delayed
import shutil
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

def load_minimum_repartiments(minimum_energy, directory, graph_folder, maps_folder):
    minimum_dfs = []
    os.makedirs(f'{ROOT}/Optimal_maps/{graph_folder}', exist_ok=True)
    seed = minimum_energy['Seed'].values
    flips = minimum_energy['Flips'].values
    for s, f in zip(seed, flips):
        carpeta = os.path.join(directory, f'{f}flips')
        fitxer = f'Repartiment_seed{int(s)}_iter10000_{int(f)}flips.pkl'
        cami = os.path.join(carpeta, fitxer)
        batch = pd.read_pickle(cami)
        minimum_dfs.append(batch)
        
        mapa_origen = os.path.join(f'{maps_folder}/{graph_folder}', f'{f}flips', f'map_seed{s}_iter10000_{f}flips.png')
        mapa_desti = os.path.join(f'{ROOT}/Optimal_maps/{graph_folder}', f'map_seed{s}_iter10000_{f}flips.png')
        if os.path.exists(mapa_origen):
            shutil.copy2(mapa_origen, mapa_desti)
    return minimum_dfs

args = arg_parse()

directory = f'{ROOT}/SynthSolutions' if args.graph == 'synthetic' else f'{ROOT}/Solutions'
maps_folder = f'{ROOT}/SynthMaps' if args.graph == 'synthetic' else f'{ROOT}/Maps'    
#Statistics of solution per value of flips/iteration

for entry in os.scandir(directory):
    if entry.is_dir():
        graph_folder = entry.name
        flip_data = pd.DataFrame()
        if args.folders is not None and graph_folder not in args.folders:
            continue
        for subentry in os.scandir(entry.path):
            if subentry.is_dir() and subentry.name:
                name = subentry.name
                folder = f'{graph_folder}_{name}'       
                batch = pd.DataFrame()
                
                repartiments = [file for file in os.listdir(subentry.path) if re.match('^Repartiment', file)] #Find filenames
                totals = [file for file in os.listdir(subentry.path) if re.match('^Totals', file)]
                repartiments = natsorted(repartiments) #It automatically sorts by seed due to naming convention of the files
                
                solucions = [pd.read_pickle(f'{subentry.path}/{sol}') for sol in repartiments] #Load dataframes
                agregats = [pd.read_pickle(f'{subentry.path}/{tot}') for tot in totals] #Load aggregate dataframes
                batch['Energy'] = [df['Desviacio'].sum() for df in agregats]
                numbers = [re.findall(r'\d+', file) for file in os.listdir(subentry.path) if re.match('^Totals', file)]
                batch['Seed'] = [int(number[0]) for number in numbers]
                batch['Flips'] = [int(number[2]) for number in numbers]
                
                flip_data = pd.concat([flip_data, batch], ignore_index=True)
        
        flip_agregat = flip_data.groupby('Flips', as_index = False).agg(Energy=('Energy' , 'mean'), Sigma=('Energy', 'sem'))
        
        if len(flip_agregat) > 1:
            flip_agregat['Significancia'] = (flip_agregat['Energy']-flip_agregat['Energy'][1])/np.sqrt(flip_agregat['Sigma']**2 + flip_agregat['Sigma'][1]**2)
        else:
            flip_agregat['Significancia'] = np.nan
        flip_data.to_pickle(f'{ROOT}/Flip_data_{graph_folder}.pkl')

        # Plotting histograms and mean energy vs flips
        
        fig, ax = plt.subplots()
        
        ax.errorbar(flip_agregat['Flips'], flip_agregat['Energy'], flip_agregat['Sigma'], fmt='o', ms=4, linewidth=2, capsize=6)
        ax.set_xlabel('Flips per iteration')
        ax.set_ylabel('Mean energy (inhabitants)')
        plt.tight_layout()
        plt.savefig(f'{ROOT}/Energy_vs_flips_{graph_folder}.png', dpi = 300)
        plt.close()  
        
        dist_folder = f'{ROOT}/Distances/{graph_folder}'
        histo_data = pd.DataFrame(columns = ['Flips', 'Mean', 'Sigma', 'Q1', 'Q3'])
        for entrada in os.scandir(dist_folder):
            if entrada.is_file():
                file = entrada.name
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
        plt.savefig(f'{ROOT}/AvgHamming_vs_flips_{graph_folder}.png', dpi = 300)
        plt.close()
        
        #Minimum energy solutions Hamming distance
        permutacions = list(permutations(range(1,8)))
        minimum_energy = flip_data[flip_data['Energy'] == flip_data['Energy'].min()]
        
        minimum_dfs = load_minimum_repartiments(minimum_energy, entry.path, graph_folder, maps_folder)
        if len(minimum_dfs) < 2:
            continue
        parells = list(combinations(range(len(minimum_dfs)), 2))
        
        resultats = Parallel(n_jobs=args.jobs)(delayed(calcul_distancia)(i, j, minimum_dfs, permutacions) for i, j in parells)
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
        filename = f'{ROOT}/Minimum_Hamming_{graph_folder}.png'
        plt.savefig(filename, dpi = 300)
        plt.close()
        