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

'''
100 seeds x 5 values of flips x 100000 iterations ~24 h 22 min
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


######################################
#Functions used to initialise the map#
######################################

def conquesta(european):
    '''
    Function that initialises the map. It plants seeds for every territory and then populates the map.
    
    Input
    ----------
    european : dataframe
        dataframe with information from every province (population, geometry, adjacencies...). 
        TERRITORY_ID = 0 for all provinces.

    Output
    -------
    european : dataframe
        populated dataframe. TERRITORY_ID column is filled with numbers 1-7.
    '''
    #Seeds are planted
    init_seeds = random.sample(range(0, 51), 7) 
    for i, seed in enumerate(init_seeds, 1):
        european.loc[seed, 'TERRITORY_ID'] = i
    
    #Map is populated
    desocupats = european.loc[european['TERRITORY_ID'] == 0] #Unoccupied provinces
    ocupats = european.loc[european['TERRITORY_ID'] != 0] #Occupied provinces
    frontera = desocupats[desocupats.index.isin(ocupats['VESINS'].explode())] #The border is defined
    while len(frontera.index) != 0: #Loop until border is exhausted
        conquerit = frontera.sample() #Picks a random unoccupied border province
        conqueridors = ocupats[ocupats.index.isin(conquerit['VESINS'].explode())] #Selects all occupied neighbours
        conqueridor = conqueridors.sample() #Picks a random occupied neighbour
        european.loc[conquerit.index[0], 'TERRITORY_ID'] = int(conqueridor['TERRITORY_ID'].iloc[0]) #Territory_ID is updated
        desocupats = european.loc[european['TERRITORY_ID'] == 0]
        ocupats = european.loc[european['TERRITORY_ID'] != 0]
        frontera = desocupats[desocupats.index.isin(ocupats['VESINS'].explode())]
        print('Unoccupied: ',len(desocupats.index))
        
    return european

##################################
#Functions used for the annealing#
##################################

def agregat(canvi):
    '''
    Function that aggregates territories and computes their deviation from the mean.

    Input
    ----------
    canvi : dataframe
        proposed change in the map

    Output
    -------
    agregat : dataframe
        Dataframe with the aggregate data for every TERRITORY_ID, including deviation from the mean

    '''
    agregat = canvi.groupby('TERRITORY_ID').agg(
        Membres=('TERRITORY_ID','size'), 
        Poblacio=('POPULATION','sum')
    )
    agregat['Desviacio'] = abs(agregat['Poblacio'] - agregat.mean().iloc[1])
    return agregat

def DFS(nodes, adj):
    '''
    Performs a DFS on a given set of nodes and adjacency matrix. Used to check contiguity
    
    Parameters
    ----------
    nodes : list
        subset of provinces involved in the DFS
    Adj : numpy array
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
        sub_adj = adj[np.ix_(nodes,nodes)] #n x n matrix
        visitats = np.zeros(n, dtype = bool)
        front = [0] #indexes have been implicitly resetted inside the sub_adj
        visitats[0] = True #starting point automatically counts as visited
        visited = 1 #visited counter
        while len(front) > 0:
            node = front.pop() #visited is removed
            veins = np.where(sub_adj[node] == 1)[0] #neighbors are selected
            for vei in veins:
                if visitats[vei] == False:
                    visitats[vei] = True #counted as visited
                    front.append(vei) #placed in the wavefront advancing through the tree
                    visited += 1
        return visited

def flipexecute(canvi):
    '''
    Implementation of the exchange of regions. It checks if the change respects contiguity.
    If not, a new exchange is proposed and so on until contiguity is maintained.

    Input
    ----------
    canvi : dataframe
       new version of the European dataframe that is going to be evaluated to see
       if the new map configuration should be accepted
       
    agregat_vell : dataframe
        dataframe with the aggregate data of the old configuration

    Output
    -------
    canvi : dataframe
        updated dataframe after the ID flip

    '''
    membres = canvi['TERRITORY_ID'].value_counts()
    preservacio = membres[membres > 1].index.tolist()    
    #territories that contain only 1 region are at risk of extinction if they are flipped and accepted
    #these territories are excludad from the list of candidates that can be flipped
    candidats_flip = canvi[canvi['TERRITORY_ID'].isin(preservacio)] #candidates to have its ID flipped
    clusters = 2
    while clusters != 1:
        candidat = candidats_flip.sample() #random candidate
        donor_nuts = candidat['NUTS_ID'].iloc[0]
        donor_id = candidat['TERRITORY_ID'].iloc[0]
        vesins_candidat = canvi[(canvi.index.isin(candidat['VESINS'].iloc[0])) & (canvi['TERRITORY_ID'] != donor_id)] #filter candidate neighbours
        if len(vesins_candidat) == 0:
            continue
        nou_id = int(vesins_candidat.sample()['TERRITORY_ID'].iloc[0]) #random pick a new ID
        #We only need to check connectivity of these two regions
        donor_region = canvi[(canvi['TERRITORY_ID'] == donor_id) & (canvi['NUTS_ID'] != donor_nuts)] #donor region without the candidate
        acceptor_region = canvi[(canvi['TERRITORY_ID'] == nou_id) | (canvi['NUTS_ID'] == donor_nuts)] #acceptor region + candidate
        donor_indices = donor_region.index.to_list()
        acceptor_indices = acceptor_region.index.to_list()
        donor_DFS = DFS(donor_indices, adj)
        acceptor_DFS = DFS(acceptor_indices, adj)
        if donor_DFS == len(donor_region) and acceptor_DFS == len(acceptor_region):
            canvi.loc[candidat.index, 'TERRITORY_ID'] = nou_id
            clusters = 1
        else:
            clusters = 2
            
    return canvi

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
    matriu_vella : dataframe
        current map configuration
    matriu_nova : dataframe
        new proposed map configuration
    temperatura : float
        temperature. A parameter that determines how often a flip that worsens the
        deviation is accepted.

    Returns
    -------
    matriu_vella : dataframe
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

def annealing(n, flips, european):
    '''
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    flips : TYPE
        DESCRIPTION.
    european : TYPE
        DESCRIPTION.

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
    
    canvi = conquesta(european)
      
    energies = []   
    mostres = 1000 #taking a few samples to estimate the appropriate initial temperature
    
    print(f'[Seed {seed}] Sampling to estimate initial temperature...')
    for _ in range(1, mostres):
        canvi_mostra = canvi.copy()
        for _ in range(1,flips+1):
            canvi_mostra = flipexecute(canvi_mostra)
        agregat_mostra = agregat(canvi_mostra)
        energies.append(float(agregat_mostra['Desviacio'].sum()))
    
    
    temp_inicial = 2*np.std(energies) #factor of 2 because I suspect exploring the landscape is difficult
    
    print(f'[Seed {seed}] Initial temperature = {temp_inicial:.0f}')
        
    x = np.array([])
    y = np.array([])
    n_iter = 10000
    
    #temp = n_iter*[temp_inicial]
    #temp = np.linspace(temp_inicial, 0, n_iter)
    temp = np.logspace(np.log10(temp_inicial), 0, n_iter)
    stored_desviacio = european['POPULATION'].sum() 
    for i in range(1,n_iter): #number of evaluated configurations
        agregat_vell = agregat(canvi)
        
        canvi_copia = canvi.copy()
        for _ in range(1, flips+1): #an arbitrary number of flips is executed before accepting or rejecting
        #this is an atempt to decorrelate the different configurations that are submitted to the metropolis algorithm
            canvi_nou = flipexecute(canvi_copia)
            
            canvi_copia = canvi_nou 
        
        canvi_nou = canvi_copia
        
        agregat_nou = agregat(canvi_nou) #new aggregate dataframe
        
        desviacio = float(agregat_vell['Desviacio'].sum()) #current deviation
        
        desviacio_nova = float(agregat_nou['Desviacio'].sum()) #new proposed deviation
    
        canvi = accepta_rebutja(desviacio_nova, desviacio, canvi, canvi_nou, temp[i-1]) #dataframe outputted by Metropolis
        
        agregat_resultat = agregat(canvi) #aggregate dataframe that has been outputted by Metropolis
        
        if float(agregat_resultat['Desviacio'].sum()) < stored_desviacio: #store the minimum value
            stored_desviacio = float(agregat_resultat['Desviacio'].sum())
            stored_agregat = agregat_resultat #Aggregate dataframe of the optimal solution
            #seleccio = canvi[['NUTS_NAME', 'TERRITORY_ID', 'POPULATION']]
            #seleccio.to_excel('Repartiment.ods')
            canvi.to_pickle(f'{sol_dir}/Repartiment_seed{seed}_iter{n_iter}_{flips}flips.pkl') #dataframe with the ID of every region. The optimal map configuration is in here
            stored_agregat.to_pickle(f'{sol_dir}/Totals_seed{seed}_iter{n_iter}_{flips}flips.pkl') #aggregate dataframe of the optimal configuration
            x = np.append(x, i)
            y = np.append(y, stored_desviacio)
            print(f'[Seed {seed}, {flips} flips] Improvement found on iteration {i} with deviation = {stored_desviacio}')
        else:
            if i % 1000 == 0: #visual check every 1000 iterations to confirm the code is running
                print(f'[Seed {seed}, {flips} flips] Iteration {i} of {n_iter}')
            
    
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
flips = [2, 4, 5, 6]
'''
10 seeds x 10000 iterations ~45 min
'''
resultats = Parallel(n_jobs=12)(delayed(annealing)(seed, flip, european.copy()) for seed in seeds for flip in flips)

end = time.time()

print('Time elapsed:', end-start)