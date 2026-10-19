#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:25:23 2026

@author: sergi
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd

def load_minimum_repartiments(minimum_energy, directory = 'Solutions'):
    minimum_dfs = []

    seed = minimum_energy['Seed'].values
    flips = minimum_energy['Flips'].values
    for s, f in zip(seed, flips):
        carpeta = os.path.join(directory, f'{f}flips')
        fitxer = f'Repartiment_seed{int(s)}_iter10000_{int(f)}flips.pkl'
        cami = os.path.join(carpeta, fitxer)
        batch = pd.read_pickle(cami)
        minimum_dfs.append(batch)
        
    return minimum_dfs

flip_data = pd.read_pickle('Flip_data.pkl')

#Minimum energy solutions Hamming distance
minimum_energy = flip_data[flip_data['Energy'] == flip_data['Energy'].min()]


minimum_dfs = load_minimum_repartiments(minimum_energy)
df = minimum_dfs[0].copy()


regions_comunes = [2, 4, 6]  


def assignar_color(territori):
    if territori in regions_comunes:
        return territori
    else:
        return 0  

df['COLOR'] = df['TERRITORY_ID'].apply(assignar_color)


n_regions = len(regions_comunes)
cmap = plt.cm.get_cmap('tab10', n_regions + 1)
colors = ['white'] + [cmap(i) for i in range(n_regions)]
custom_cmap = mcolors.ListedColormap(colors)


fig, ax = plt.subplots(1, 1, figsize=(12, 10))
df.plot(column='COLOR', 
        cmap=custom_cmap, 
        ax=ax, 
        edgecolor='grey', 
        linewidth=0.5,
        legend=False)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('regions_comunes.png', dpi=300, bbox_inches='tight')
plt.show()