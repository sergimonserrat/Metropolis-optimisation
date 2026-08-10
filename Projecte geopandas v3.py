# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:46:21 2024

@author: Sergi
"""
from libpysal.weights import W, Rook, KNN
import matplotlib.pyplot as plt
import networkx as nx
import geopandas
import numpy as np
import random
import pandas

seed = 2
np.random.seed(seed)
random.seed(seed)

#the pkl file contains all the neighbour-node relationships encoded in the VESINS column
european = pandas.read_pickle(r'/home/sergi/Documents/Dataframe_previ.pkl') #this path works for me, likely won't work for you

#this libpysal part is only used to plot the graph (plot 3)
#the annealing part builds the graph using the pkl file
centroids = np.column_stack((european.centroid.x, european.centroid.y))

w_rook = Rook.from_dataframe(european, use_index = False)

w_knn3 = KNN.from_dataframe(european, k=3)
'''
european['TERRITORY_ID'] = 0
european['VESINS'] = ''
for key in w_attach.neighbors:
    european['VESINS'].iloc[key] = w_attach.neighbors[key]

european.to_pickle('Dataframe_previ.pkl')'''

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

######################################
#Functions used to initialise the map#
######################################

def plant_seeds(european):
    '''
    This function takes the column TERRITORY_ID and assignes a non-zero value to each region (row)
    whilst respecting contiguity. 7 seeds are generated and then spread out by checking 
    their empty (TERRITORY_ID = 0) neighbours and assigning them the same ID as the neighbouring seed.
    
    Take a random seed with TERRITORY_ID = N --> choose random empty neighbour --> change TERRITORY_ID from 0 to N
    Repeat until no empty regions remain.
    
    Input
    ----------
    european : dataframe
        contains all the information on regions, populations, neighbours...
        see Dataframe_previ.pkl metadata

    Output
    -------
    european : dataframe
        the input dataframe with a TERRITORY_ID assigned to each region
    '''
    init_seeds = random.sample(range(0, 51), 7)
    fronteres = np.array([])
    i = 1
    for seed in init_seeds:
        european.loc[seed, 'TERRITORY_ID'] = i
        frontera = w_attach.neighbors[seed]
        fronteres = np.append(fronteres, frontera)
        i = i+1
    #seeds for every region have been generated
    return european   

def ocupacio(european):
    '''
    Function that returns two dataframes. One for occupied regions and another for unoccupied regions
    
    Input
    ----------
    european : dataframe
        contains all the information on regions, populations, neighbours...
        see Dataframe_previ.pkl metadata

    Output
    -------
    ocupats : dataframe
        dataframe with all the occupied regions
    desocupats : dataframe
        dataframe with all the unoccupied regions
    '''
    desocupats = european.loc[european['TERRITORY_ID'] == 0] #Territoris sense ocupar
    ocupats = european.loc[european['TERRITORY_ID'] != 0] #Territoris ocupats
    return ocupats, desocupats

def conquesta(ocupats, desocupats):
    '''
    Function used to populate the map after planting the seeds
    
    Input
    ----------
    ocupats : dataframe
        dataframe with occupied region
    desocupats : dataframe
        dataframe with unoccuipied regions

    Output
    -------
    conqueridor : dataframe
        region that is picked as occupier
    conquereix : dataframe
        randomly picked neighboring unoccupied region
    '''
    conqueridor = ocupats.sample()
    vol_conquerir = desocupats[desocupats.index.isin(conqueridor['VESINS'].tolist()[0])] 
    #picks neighbours
    if len(vol_conquerir.index) == 0:
        conquereix = conqueridor
    else:
        conquereix = vol_conquerir.sample() #random neighbour
    return conqueridor, conquereix

def annexio(conqueridor, conquereix, european):
    '''
    Function that implements the change in TERRITORY_ID in the European dataframe
    once a conquest/annexation has been accepted

    Parameters
    ----------
    conqueridor : dataframe
        region that occupies
    conquereix : dataframe
        region that is occupied
    european : dataframe
        intentary of all regions

    Returns
    -------
    european : dataframe
        dataframe updated with the occupation

    '''
    european.loc[conquereix.index[0], 'TERRITORY_ID'] = int(conqueridor['TERRITORY_ID'].iloc[0])
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

def flipexecute(canvi, agregat_vell):
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
    preservacio = agregat_vell[agregat_vell['Membres'] > 1].index.tolist()
    #territories that contain only 1 region are at risk of extinction if they are flipped and accepted
    #these territories are excludad from the list of candidates that can be flipped

    candidats_flip = canvi[canvi['TERRITORY_ID'].isin(preservacio)] #candidates to have its ID flipped
    
    clusters = 2
    while clusters != 1:
        candidat = candidats_flip.sample() #random candidate
        candidat_id = candidat['TERRITORY_ID'].iloc[0]
        candidat_nuts = candidat['NUTS_ID'].iloc[0]
        
        contiguitat = canvi[(canvi['TERRITORY_ID'] == candidat_id) & 
                           (canvi['NUTS_ID'] != candidat_nuts)]
        
        d = {}
        for row in contiguitat.index:
            values = canvi.loc[row, 'VESINS']
            d[row] = [valor for valor in values if valor in contiguitat.index]
        
        graf = nx.from_dict_of_lists(d) #a graph of the territory without the candidate is constructed
        try:
            if nx.is_connected(graf): #contiguity check 
                clusters = 1 #only get out of the loop if the territory maintains contiguity after the flip
        except nx.NetworkXPointlessConcept: #error management
            clusters = 2
    
    #I should touch this part to exclude neighbours that share ID with the candidate, otherwise I am doing pointless flips
    vesins_candidat = canvi[canvi.index.isin(candidat['VESINS'].iloc[0])] #filter candidate neighbours
    nou_id = int(vesins_candidat.sample()['TERRITORY_ID'].iloc[0]) #random pick a new ID
    canvi.loc[candidat.index, 'TERRITORY_ID'] = nou_id
    
    return canvi

'''
def genera_graf(dataframe):
    d = {}
    for row in dataframe.index:
        d[row] = dataframe.at['VESINS'].iloc[row]
    graf = W(d)
    return d, graf
'''

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

'''
def avaluacio_canvi(agregat_vell, agregat_nou):
    distancia_canvi = agregat_nou - agregat_vell
    return distancia_canvi
'''
w_attach = attach_islands(w_rook, w_knn3)

dataframe = plant_seeds(european)
while european['TERRITORY_ID'].min() == 0:


    ocupats, desocupats = ocupacio(european)
    if len(desocupats.index) == 1:
       conquereix = desocupats
       conqueridors = european[european.index.isin(conquereix['VESINS'].tolist()[0])]
       conqueridor = conqueridors.sample()
    else:   
        conqueridor, conquereix = conquesta(ocupats, desocupats)
   
    canvi = annexio(conqueridor, conquereix, european)
        
    print('Unoccupied: ',len(desocupats.index))
  
energies = []   
mostres = 1000 #taking a few samples to estimate the appropriate initial temperature

for _ in range(1, mostres):
    canvi_mostra = canvi.copy()
    for _ in range(1,5):
        agregat_mostra =agregat(canvi_mostra)
        canvi_mostra = flipexecute(canvi_mostra, agregat_mostra)
    agregat_mostra = agregat(canvi_mostra)
    energies.append(float(agregat_mostra['Desviacio'].sum()))


temp_inicial = 2*np.std(energies) #factor of 2 because I suspect exploring the landscape is difficult

print(f'temp_inicial = {temp_inicial:.0f}')
    
x = np.array([])
y = np.array([])
n_iter = 10000

#temp = n_iter*[temp_inicial]
#temp = np.linspace(temp_inicial, 0, n_iter)
temp = np.logspace(np.log10(temp_inicial), 0, n_iter)
stored_desviacio = european['POPULATION'].sum() 
for i in range(1,n_iter): #number of evaluated configurations
    '''
    approximate running times (intel core i5):
        1000 iterations ~30 s
        10000 iterations ~4 min
        100000 iterations ~45 min
    Constant schedule: few improvements after 2000-3000 iterations.
    Linear schedule: improvements concentrated at the end. Worse solutions than constant
    schedule. Inadequate initial T? What if linear schedule was a series of constant schedules with 3000 iterations for each 
    temperature value?
    Exponential schedule: improvements concentrated within the first few thousand iterations
    '''
    agregat_vell = agregat(canvi)
    
    canvi_copia = canvi.copy()
    for _ in range(1, 5): #an arbitrary number of flips is executed before accepting or rejecting
    #this is an atempt to decorrelate the different configurations that are submitted to the metropolis algorithm
        canvi_nou = flipexecute(canvi_copia, agregat_vell)
        
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
        seleccio = canvi[['NUTS_NAME', 'TERRITORY_ID', 'POPULATION']]
        seleccio.to_excel('Repartiment.ods')
        canvi.to_pickle('Repartiment.pkl') #dataframe with the ID of every region. The optimal map configuration is in here
        stored_agregat.to_pickle('Totals.pkl') #aggregate dataframe of the optimal configuration
        x = np.append(x, i)
        y = np.append(y, stored_desviacio)
        print('Improvement found on interation ', i)
    else:
        if i % 1000 == 0: #visual check every 1000 iterations to confirm the code is running
            print('Currently on iteration ', i)
        

plt.figure()
# Subplot posició x
plt.grid(True)
plt.title(f'Convergence. seed = {seed} iterations = {n_iter} schedule = exponential')
plt.xlabel('Iteration number')
plt.ylabel('Sum of deviations (inhabitants)')
plt.plot(x, y)
plt.tight_layout()
filename = f'convergence_seed{seed}_iter{n_iter}_exp.png'
plt.savefig(filename)
plt.close()

european = pandas.read_pickle('Repartiment.pkl')

ax = european.plot(column = 'TERRITORY_ID') #plots the 7 regions

ax.set_axis_off()
ax.set_title(f'Map configuration. seed = {seed} iterations = {n_iter}')
filename2 = f'map_seed{seed}_iter{n_iter}_exp.png'
plt.savefig(filename2, dpi = 300, bbox_inches='tight')
plt.close()

ax = european.plot(edgecolor='grey', facecolor='w')
f,ax = w_attach.plot(european, ax=ax,
        edge_kws=dict(color='r', linestyle=':', linewidth=1),
        node_kws=dict(marker=''))
ax.set_axis_off() #plots the underlying graph
filename3 = 'graph.png'
plt.savefig(filename3, dpi = 300, bbox_inches='tight')
plt.close()
