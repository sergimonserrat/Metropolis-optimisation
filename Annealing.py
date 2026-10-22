# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:46:21 2024

@author: Sergi Monserrat Mascaró
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from libpysal.weights import Rook, KNN
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pandas
from joblib import Parallel, delayed
import time
#import cProfile
#import pstats
from numba import njit

'''
100 seeds x 5 values of flips x 100000 iterations ~1 h 27 min
'''

start = time.time()
#the pkl file contains all the neighbour-node relationships encoded in the VESINS column
european = pandas.read_pickle(r'/home/sergi/Documents/Dataframe_previ.pkl') #this path works for me, likely won't work for you

#this libpysal part is only used to plot graph.png
#the annealing part builds the graph using the pkl file
centroids = np.column_stack((european.centroid.x, european.centroid.y))
    
w_rook = Rook.from_dataframe(european, use_index = False)
    
w_knn3 = KNN.from_dataframe(european, k=3)

ax = european.plot(edgecolor='grey', facecolor='w')

#personalised attach islands since the command attach_island only allows knn1 connection
def attach_islands(w_rook, w_knn3): 
    illes = []
    for regio in w_rook.neighbors:
        if len(w_rook.neighbors[regio]) == 0:
            illes.append(regio)
    
    for illa in illes:
        veins_knn = w_knn3.neighbors[illa]
        w_rook.neighbors[illa] = veins_knn
        w_rook.weights[illa] = [1.0] * len(veins_knn)
    
    w_rook.symmetrize(inplace=True) #just in case. I had a bad experience with an asymmetric graph
    return w_rook

w_attach = attach_islands(w_rook, w_knn3)
f,ax = w_attach.plot(european, ax=ax,
        edge_kws=dict(color='r', linestyle=':', linewidth=1),
        node_kws=dict(marker=''))
ax.set_axis_off() #plots the underlying graph
filename3 = 'graph.png'
plt.savefig(filename3, dpi = 300, bbox_inches='tight')
plt.close()

graph = w_attach.to_networkx()

# Extreure la matriu d'adjacències
adj = nx.to_numpy_array(graph)
populations = np.array(european['POPULATION'])
territory_id = np.array(european['TERRITORY_ID'])

######################################
#Functions used to initialise the map#
######################################

def conquesta(adj, territory_id):
    '''
    Function that initialises the map. It plants seeds for every territory and then populates the map.
    
    Input
    ----------
    adj : array
        adjacency matrix
        
    territory_id : array
        array with all elements set to zero

    Output
    -------
    territory_id : array
        populated array with numbers 1-7.
    '''
    #Seeds are planted
    init_seeds = random.sample(range(0, 51), 7) 
    for i, seed in enumerate(init_seeds, 1):
        territory_id[seed] = i
    
    #Map is populated
    desocupats = (territory_id == 0)
    ocupats = (territory_id != 0)
    frontera_mask = (desocupats & (adj @ ocupats > 0))
    frontera = np.where(frontera_mask)[0]
    while frontera_mask.any():
        conquerit = np.random.choice(frontera)
        conqueridors = np.where((territory_id != 0) & (adj[conquerit] == 1))[0]
        conqueridor = np.random.choice(conqueridors)
        territory_id[conquerit] = territory_id[conqueridor]
        
        frontera_mask[conquerit] = False
        veins_conquerit = np.where((adj[conquerit] == 1) & (territory_id == 0))[0]
        frontera_mask[veins_conquerit] = True
        frontera = np.where(frontera_mask)[0]

    return territory_id

##################################
#Functions used for the annealing#
##################################

def agregat(territory_id, populations):
    '''
    Function that aggregates territories and computes their deviation from the mean.

    Input
    ----------
    territory_id : array
        proposed change in the map

    Output
    -------
    agregat_membres : array
        member count for each region
    agregat_poblacio : array
        aggregated population for each region
    desviacio : float
        energy. Deviation from the mean.

    '''
    agregat_membres = np.bincount(territory_id)[1:]
    agregat_poblacio = np.bincount(territory_id, weights = populations)[1:]
    desviacio = abs(agregat_poblacio - np.mean(agregat_poblacio))
    return agregat_membres, agregat_poblacio, desviacio

@njit
def DFS(nodes, adj):
    '''
    Performs a DFS on a given set of nodes and adjacency matrix. Used to check contiguity
    
    Parameters
    ----------
    nodes : list
        subset of provinces involved in the DFS
    adj : numpy array
        matrix of adjacencies of the selected nodes

    Returns
    -------
    visitats : int
        number of visited nodes

    '''
    n = len(nodes)
    if n <= 1:
        return n
    else:
        #sub_adj = adj[np.ix_(nodes,nodes)] #n x n matrix
        sub_adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sub_adj[i, j] = adj[nodes[i], nodes[j]]
        visitats = np.zeros(n, dtype = np.bool_)
        front = np.zeros(n, dtype = np.int64) #indexes have been implicitly resetted inside the sub_adj
        pile = 1
        visitats[0] = True #starting point automatically counts as visited
        visited = 1 #visited counter
        while pile > 0:
            pile -= 1
            node = front[pile]
            veins = np.where(sub_adj[node] == 1)[0] #neighbors are selected
            for vei in veins:
                if visitats[vei] == False:
                    visitats[vei] = True #counted as visited
                    front[pile] = vei
                    pile += 1
                    visited += 1
        return visited

def flipexecute(territory_id, adj, agregat_membres):
    '''
    Implementation of the exchange of regions. It checks if the change respects contiguity.
    If not, a new exchange is proposed and so on until contiguity is maintained.

    Input
    ----------
    agregat_membres: array
        member count for each region
        
    adj : array
        adjacency matrix
    
    territory_id : array
        map configuration
    Output
    -------
    territory_id : array
        updated map configuration after the flip

    '''
    preservacio = np.where(agregat_membres > 1)[0] + 1
    candidats_flip = np.where(np.isin(territory_id, preservacio))[0]
    clusters = 2
    while clusters !=1:
        candidat = np.random.choice(candidats_flip)
        vesins_candidat = np.where((territory_id != territory_id[candidat]) & (adj[candidat] == 1))[0]
        if len(vesins_candidat) == 0:
            continue
        nou_id = int(np.random.choice(territory_id[vesins_candidat]))
        donor_region = np.where(territory_id == territory_id[candidat])[0]
        donor_region = donor_region[donor_region != candidat]
        acceptor_region = np.where(territory_id == nou_id)[0]
        acceptor_region = np.append(acceptor_region, candidat)
        donor_DFS = DFS(donor_region, adj)
        acceptor_DFS = DFS(acceptor_region, adj)
        if donor_DFS == len(donor_region) and acceptor_DFS == len(acceptor_region):
            territory_id[candidat] = nou_id
            clusters = 1
        else:
            clusters = 2
            
    return territory_id

def accepta_rebutja(desviacio_nova, desviacio, 
                    matriu_vella, matriu_nova, temperatura):
    '''
    Implementation of the Metropolis algorithm.

    Parameters
    ----------
    desviacio_nova : float
        new proposed deviation from the mean
    desviacio : float
        current deviation from the mean
    matriu_vella : array
        current map configuration
    matriu_nova : array
        new proposed map configuration
    temperatura : float
        temperature. A parameter that determines how often a flip that worsens the
        deviation is accepted.

    Returns
    -------
    matriu_vella : array
        updated dataframe after the Metropolis algorithm

    '''
    if desviacio_nova < desviacio:
        matriu_vella = matriu_nova #accept improvements
    else:
        empitjorament = desviacio_nova - desviacio
        llindar = np.exp(-(empitjorament/temperatura)) 
        rebuig = np.random.rand() #accept worsening with a certain probability
        if llindar < rebuig:
            pass
        else:
            matriu_vella = matriu_nova
    return matriu_vella


directory = 'Solutions'
os.makedirs(directory, exist_ok = True)
print(f'Solution data will be stored in {directory} folder')

maps_folder = 'Maps'
os.makedirs(maps_folder, exist_ok = True)
print(f'Maps will be stored in {maps_folder} folder')

convergence_folder = 'Convergence'
os.makedirs(convergence_folder, exist_ok = True)
print(f'Convergence plots will be stored in {convergence_folder} folder')

##########################
#Annealing implementation#
##########################

def annealing(n, flips, adj, territory_id, populations, european):
    '''
    Main programs. Performs the annealing
    
    Parameters
    ----------
    n : integer
        seed number
    flips : integer
        number of flips per iteration
    territory_id : array
        map configuration
    populations : array
        population per province
    european : dataframe
        starting dataframe containing all the information on adjacencies, populations, etc.
        Here it is only called to update the TERRITORY_ID column after the annealing so
        everything can be stored into a .pkl file.
        
    Returns
    -------
    None.

    '''
    sol_dir = f'{directory}/{flips}flips'
    map_dir = f'{maps_folder}/{flips}flips'
    conver_dir = f'{convergence_folder}/{flips}flips'
    os.makedirs(sol_dir, exist_ok = True)
    os.makedirs(map_dir, exist_ok = True)
    os.makedirs(conver_dir, exist_ok = True)
    seed = n
    np.random.seed(seed)
    random.seed(seed)
    
    canvi = conquesta(adj, territory_id)
      
    energies = []   
    mostres = 1000 #taking a few samples to estimate the appropriate initial temperature
    
    print(f'[Seed {seed}] Sampling to estimate initial temperature...')
    agregat_mostra = agregat(canvi, populations)
    for _ in range(1, mostres):
        canvi_mostra = canvi.copy()
        for _ in range(1,flips+1):
            canvi_mostra = flipexecute(canvi_mostra, adj, agregat_mostra[0])
        agregat_mostra = agregat(canvi_mostra, populations)
        energies.append(float(np.sum(agregat_mostra[2])))
    
    
    temp_inicial = 2*np.std(energies) #factor of 2 because I suspect exploring the landscape is difficult
    
    print(f'[Seed {seed}] Initial temperature = {temp_inicial:.0f}')
        
    x = np.array([])
    y = np.array([])
    n_iter = 10000
    
    #temp = n_iter*[temp_inicial]
    #temp = np.linspace(temp_inicial, 0, n_iter)
    temp = np.logspace(np.log10(temp_inicial), 0, n_iter)
    stored_desviacio = np.sum(populations)
    repartiment = territory_id.copy()
    for i in range(1,n_iter): #number of evaluated configurations
        agregat_vell = agregat(canvi, populations)
        
        canvi_copia = canvi.copy()
        agregat_membres = agregat_vell[0]
        for _ in range(1, flips+1): #an arbitrary number of flips is executed before accepting or rejecting
        #this is an atempt to decorrelate the different configurations that are submitted to the metropolis algorithm
            canvi_nou = flipexecute(canvi_copia, adj, agregat_membres)
            
            canvi_copia = canvi_nou
            
            agregat_membres = agregat(canvi_nou, populations)[0]
        
        canvi_nou = canvi_copia
        
        agregat_nou = agregat(canvi_nou, populations)
        
        desviacio = float(np.sum(agregat_vell[2])) #current deviation
        
        desviacio_nova = float(np.sum(agregat_nou[2])) #new proposed deviation
    
        canvi = accepta_rebutja(desviacio_nova, desviacio, canvi, canvi_nou, temp[i-1]) #dataframe outputted by Metropolis
        
        agregat_resultat = agregat(canvi, populations) #aggregate dataframe that has been outputted by Metropolis
        
        if float(np.sum(agregat_resultat[2])) < stored_desviacio: #store the minimum value
            stored_desviacio = float(np.sum(agregat_resultat[2]))
            stored_agregat = agregat_resultat #Aggregate dataframe of the optimal solution
            repartiment = canvi
            x = np.append(x, i)
            y = np.append(y, stored_desviacio)
            print(f'[Seed {seed}, {flips} flips] Improvement found on iteration {i} with deviation = {stored_desviacio}')
        else:
            if i % 1000 == 0: #visual check every 1000 iterations to confirm the code is running
                print(f'[Seed {seed}, {flips} flips] Iteration {i} of {n_iter}')
    
    european['TERRITORY_ID'] = repartiment
    european.to_pickle(f'{sol_dir}/Repartiment_seed{seed}_iter{n_iter}_{flips}flips.pkl') #dataframe with the ID of every region. The optimal map configuration is in here
    totals = pandas.DataFrame({'TERRITORY_ID': range(1, 8), 'Membres': stored_agregat[0], 'Poblacio': stored_agregat[1],
                                   'Desviacio': stored_agregat[2]})
    totals.to_pickle(f'{sol_dir}/Totals_seed{seed}_iter{n_iter}_{flips}flips.pkl')
            
    plt.figure()
    # Subplot posició x
    plt.grid(True)
    plt.title('Convergence')
    plt.xlabel('Iteration number')
    plt.ylabel('Sum of deviations (inhabitants)')
    plt.plot(x, y)
    plt.tight_layout()
    filename = f'{conver_dir}/convergence_seed{seed}_iter{n_iter}_ {flips}flips.png'
    plt.savefig(filename)
    plt.close()
    
    solucio = pandas.read_pickle(f'{directory}/{flips}flips/Repartiment_seed{seed}_iter{n_iter}_{flips}flips.pkl')
    
    ax = solucio.plot(column = 'TERRITORY_ID') #plots the 7 regions
    
    ax.set_axis_off()
    filename2 = f'{map_dir}/map_seed{seed}_iter{n_iter}_{flips}flips.png'
    plt.savefig(filename2, dpi = 300, bbox_inches='tight')
    plt.close()
    
seeds = range(0,100)
flips = [1, 3, 7, 15, 25]

resultats = Parallel(n_jobs=12)(delayed(annealing)(seed, flip, adj.copy(), territory_id.copy(), populations.copy(), european.copy()) for seed in seeds for flip in flips)
'''
with cProfile.Profile() as pr:
    annealing(0, 3, adj, territory_id, populations, european)

stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(20)
'''
end = time.time()
print('Time elapsed:', end-start)