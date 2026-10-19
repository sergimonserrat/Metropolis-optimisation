#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:34:22 2026

@author: sergi
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

start = time.time()
#the pkl file contains all the neighbour-node relationships encoded in the VESINS column
european = pandas.read_pickle(r'/home/sergi/Documents/Dataframe_previ.pkl') #this path works for me, likely won't work for you
    
w_rook = Rook.from_dataframe(european, use_index = False)
    
w_knn3 = KNN.from_dataframe(european, k=3)

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

graph = w_attach.to_networkx()
#graph.remove_edge(40,43)

average = np.sum([i*nx.degree_histogram(graph)[i] for i in range(len(nx.degree_histogram(graph)))])/np.sum(nx.degree_histogram(graph))

print('Number of nodes:', nx.number_of_nodes(graph))
print('Number of edges:', nx.number_of_edges(graph))
print('Density:', nx.density(graph))
print('Average degree:', average)
print('Planarity:', nx.check_planarity(graph))
print('Diameter:', nx.diameter(graph))
print('Avg shortest path:', nx.average_shortest_path_length(graph))
print('Clustering coefficient:', nx.average_clustering(graph))
print('Maximum degree:', len(nx.degree_histogram(graph))-1)
print('Degree histogram:', nx.degree_histogram(graph))

from scipy.stats import percentileofscore

def small_world_test_random_degree(G_obs, n_bootstrap=100):
    """
    Test d'intersecció de Lovekar et al. usant random_degree_sequence_graph.
    Retorna també el percentil de G_obs respecte a la distribució nul·la.
    """
    # 1. Mètriques del graf observat
    C_obs = nx.average_clustering(G_obs)
    L_obs = nx.average_shortest_path_length(G_obs)
    
    # 2. Seqüència de graus
    degree_seq = [d for n, d in G_obs.degree()]
    
    # 3. Generar distribució nul·la
    def generar_nul(_):
        try:
            G_rand = nx.random_degree_sequence_graph(degree_seq, seed=None)
            if nx.is_connected(G_rand):
                return nx.average_clustering(G_rand), nx.average_shortest_path_length(G_rand)
        except:
            pass
        return None, None

    resultats = Parallel(n_jobs=12)(delayed(generar_nul)(i) for i in range(n_bootstrap))
    
    C_nul = np.array([c for c, l in resultats if c is not None])
    L_nul = np.array([l for c, l in resultats if l is not None])
    
    # 4. Percentils de G_obs
    # Percentil de clustering: com de gran és C_obs respecte a la nul·la
    percentil_C = percentileofscore(C_nul, C_obs, kind='mean')
    
    # Percentil de camí mitjà: com de petit és L_obs respecte a la nul·la
    # (invertim perquè un L petit és "bo" per a món petit)
    percentil_L = percentileofscore(L_nul, L_obs, kind='mean')
    
    # 5. Cutoffs (segons el paper)
    K_C = np.percentile(C_nul, 99)
    K_L = np.percentile(L_nul, 99)
    
    # 6. Decisió
    es_small_world = (C_obs > K_C) and (L_obs < K_L)
    
    # 7. Report
    print(f"--- Clustering ---")
    print(f"C_obs = {C_obs:.4f}")
    print(f"Null: mitjana = {np.mean(C_nul):.4f}, std = {np.std(C_nul):.4f}")
    print(f"K_C (99%): {K_C:.4f}")
    print(f"Percentil de C_obs: {percentil_C:.1f}%")
    print()
    print(f"--- Camí mitjà ---")
    print(f"L_obs = {L_obs:.4f}")
    print(f"Null: mitjana = {np.mean(L_nul):.4f}, std = {np.std(L_nul):.4f}")
    print(f"K_L (99%): {K_L:.4f}")
    print(f"Percentil de L_obs (invertit): {percentil_L:.1f}%")
    print()
    print(f"--- Decisió ---")
    print(f"Condició 1 (C_obs > K_C): {'✅' if C_obs > K_C else '❌'}")
    print(f"Condició 2 (L_obs < K_L): {'✅' if L_obs < K_L else '❌'}")
    print(f"Resultat: {'SÍ' if es_small_world else 'NO'} és de món petit (estadísticament)")
    
    return {
        'es_small_world': es_small_world,
        'C_obs': C_obs,
        'L_obs': L_obs,
        'K_C': K_C,
        'K_L': K_L,
        'percentil_C': percentil_C,
        'percentil_L': percentil_L,
        'C_null': C_nul,
        'L_null': L_nul,
    }
    
    return es_small_world, K_C, K_L

degree_hist = nx.degree_histogram(graph)
degree_sequence = []
for degree, count in enumerate(degree_hist):
    degree_sequence.extend([degree] * count)

small_world_test_random_degree(graph, 10000)
aleatori = nx.random_degree_sequence_graph(degree_sequence)

print('Number of nodes:', nx.number_of_nodes(aleatori))
print('Number of edges:', nx.number_of_edges(aleatori))
print('Density:', nx.density(aleatori))
print('Average degree:', average)
print('Planarity:', nx.check_planarity(aleatori))
print('Diameter:', nx.diameter(aleatori))
print('Avg shortest path:', nx.average_shortest_path_length(aleatori))
print('Clustering coefficient:', nx.average_clustering(aleatori))
print('Maximum degree:', len(nx.degree_histogram(aleatori))-1)
print('Degree histogram:', nx.degree_histogram(aleatori))