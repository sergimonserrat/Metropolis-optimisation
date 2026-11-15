#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:49:23 2026

@author: sergi
"""

import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from libpysal.weights import Rook, KNN
from libpysal.cg import voronoi_frames
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas
import pymc
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph',
        choices=['synthetic', 'real'],
        default='synthetic',
        help='Type of graph to partition (default: synthetic).',
    )
    parser.add_argument(
        '--n_graphs',
        type=int,
        default=20,
        help='Number of synthetic graphs generated (default: 20).',
    )
    return parser.parse_args()

args = arg_parse()
if args.graph == 'synthetic':
    directory = '{ROOT}/Synthetic_graphs'
    os.makedirs(directory, exist_ok = True)
    
    comptador = 0
    synthgraphs = []
    while len(synthgraphs) < args.n_graphs:
        n_nodes = 51
        np.random.seed(comptador)
        nodes = np.random.rand(n_nodes, 2)
        
        cells, vertices = voronoi_frames(nodes, as_gdf=True, return_input=True)
    
        synth = cells.join(vertices, rsuffix="_point")
        poblacions = np.random.lognormal(mean=13.24, sigma=1.01, size=n_nodes)
        synth['POPULATION'] = poblacions
        synth['TERRITORY_ID'] = 0
        if poblacions.sum()/poblacions.max() < 7:
            comptador += 1
            continue
        w_rook = Rook.from_dataframe(cells)
        
        graf = w_rook.to_networkx()
        
        adj = nx.to_numpy_array(graf)
        
        
        average = np.sum([i*nx.degree_histogram(graf)[i] for i in range(len(nx.degree_histogram(graf)))])/np.sum(nx.degree_histogram(graf))
        synthgraphs.append(synth)
        synth.to_pickle(f'{directory}/Synthetic{comptador}.pkl')
        comptador += 1
        
        print('[Synthetic{comptador}] Number of nodes:', nx.number_of_nodes(graf))
        print('[Synthetic{comptador}] Number of edges:', nx.number_of_edges(graf))
        print('[Synthetic{comptador}] Density:', nx.density(graf))
        print('[Synthetic{comptador}] Average degree:', average)
        print('[Synthetic{comptador}] Planarity:', nx.check_planarity(graf))
        print('[Synthetic{comptador}] Diameter:', nx.diameter(graf))
        print('[Synthetic{comptador}] Avg shortest path:', nx.average_shortest_path_length(graf))
        print('[Synthetic{comptador}] Clustering coefficient:', nx.average_clustering(graf))
        print('[Synthetic{comptador}] Maximum degree:', len(nx.degree_histogram(graf))-1)
        print('[Synthetic{comptador}] Degree histogram:', nx.degree_histogram(graf))
elif args.graph == 'real':
    file = f'{ROOT}/Dataframe_previ.pkl'

    #the pkl file contains all the neighbour-node relationships encoded in the VESINS column
    european = pandas.read_pickle(file) #this path works for me, likely won't work for you
        
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
    
    degree_hist = nx.degree_histogram(graph)
    degree_sequence = []
    for degree, count in enumerate(degree_hist):
        degree_sequence.extend([degree] * count)

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

    populations = np.array(european['POPULATION'])
    pop_powerlaw = np.sort(populations)[::-1]

    #CCDF
    n = len(populations)
    ccdf = np.arange(1, n + 1) / n

    plt.figure()
    plt.loglog(pop_powerlaw, ccdf, 'o', markersize=4)
    plt.xlabel('Population')
    plt.ylabel('P(X > population)')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'{ROOT}/ccdf_poblacions.png', dpi=300)
    plt.show()
    plt.close()
    log_population = np.log(populations)

    with pymc.Model() as model:
        mu = pymc.Normal('mu', mu = np.mean(log_population), sigma = 2)
        sigma = pymc.HalfNormal('sigma', sigma = 2)
        
        likelihood = pymc.LogNormal('likelihood', mu=mu, sigma=sigma, observed=populations)
        MCMC = pymc.sample(10000, tune = 10000, return_inferencedata = True)


    mu_samples = MCMC.posterior['mu'].values.flatten()
    sigma_samples = MCMC.posterior['sigma'].values.flatten()

    mu_mean = np.mean(mu_samples)
    mu_lower = np.percentile(mu_samples, 2.5)
    mu_upper = np.percentile(mu_samples, 97.5)

    sigma_mean = np.mean(sigma_samples)
    sigma_lower = np.percentile(sigma_samples, 2.5)
    sigma_upper = np.percentile(sigma_samples, 97.5)

    print(f'Mu: {mu_mean:.2f} [95% CI: {mu_lower:.2f}, {mu_upper:.2f}]')
    print(f'Sigma: {sigma_mean:.2f} [95% CI: {sigma_lower:.2f}, {sigma_upper:.2f}]')

    #Plot posteriors
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #Mu posterior
    axes[0].hist(mu_samples, bins=30, density=True)
    axes[0].axvline(mu_mean, color='black', linestyle='--')
    axes[0].set_xlabel(r'$\mu$')
    axes[0].set_ylabel('Density')

    #Sigma posterior
    axes[1].hist(sigma_samples, bins=30, density=True)
    axes[1].axvline(sigma_mean, color='black', linestyle='--')
    axes[1].set_xlabel(r'$\sigma$')
    axes[1].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(f'{ROOT}/posterior_lognormal.png', dpi=300)
    plt.show()
    plt.close()
    print(f'Posterior mu: {MCMC.posterior['mu'].mean().values:.2f} ± {MCMC.posterior['mu'].std().values:.2f}')
    print(f'Posterior sigma: {MCMC.posterior['sigma'].mean().values:.2f} ± {MCMC.posterior['sigma'].std().values:.2f}')