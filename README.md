# Metropolis-optimisation
A coding project that groups the provinces of Spain into 7 territories of equal population. Using the Metropolis algorithm, it divides a provincial map into 7 contiguous regions by minimising the deviation from the mean population.

# Repository structure:
- Projecte geopandas v3.py: Main code
- Dataframe_previ.pkl: Graph and population data. 

| Column | Type | Description |
|---|---|---|
| NUTS_ID | string | EU NUTS-3 Identifier |
| NUTS_NAME | string | Province name |
| geometry | multipolygon | Geometry of the province |
| CENTROID | point | Coordinates of the centre of the province |
| TERRITORY_ID | integer | ID number (1-7) of the territory the province belongs to |
| POPULATION | integer | Population of the province |
| VESINS | list | Neighbouring provinces. Adjacency list of the graph |

- Requirements.txt: list of necessary libraries and their versions
- README.md: this file

# Code description 
- Create a graph with rook connectivity using libpysal.
- Attach islands using 3 nearest neighbours. (Only used to draw the graph, not in the optimisation)
- Plant seeds for each territory and expand them until they occupy the map. (Expansion of seeds is currently random. Works good enough for 50 nodes, but it will not when I try to scale up to all EU NUTS-3 regions) 
- Flips are proposed. Connectivity is check to ensure the regions are contiguous. Metropolis algorithm is implemented with a linear temperature decrease schedule. (try exponential schedule?)
- Best map configuration is saved as `Repartiment.pkl` and the aggregate data for each TERRITORY_ID is stored in `Totals.pkl`
- Create 3 plots: one for the graph, one for the map configuration, one for the convergence.

# Results and conclusions
Convergence is fast for the first 2000-3000 iterations, then stalls (diminishing returns). Keeping the temperature constant or decreasing it linearly does not change the results significantly. Different runs yield different map configurations that appear qualitatively distinct (a definition of distance in the space of map configurations would be needed). The fact that a "good" solution is easy to achieve but the optimal one is difficult to find and lots of them do not "look alike" makes me think that the space of solution might look either like "gravel pavement" (flat at a large scale but full of shallow local minima) or "karstic" (deep wells separated by almost vertical walls. A spiky landscape).

# Future improvements
In its current form the code has deficiencies for what it is already trying to do. These are some things this project could do better:

- Random seed management. The biggest limitation to studying the results is that the code does not pick a seed for the randomly generated numbers. There is no reproducibility.
- Build the graph and dataframe in situ. Right now, the code takes a prebuilt dataframe that I created by manually adding a few adjacencies to a graph that was automatically built with networkx. My intention to scale up to all EU NUTS-3 regions would require adding many more manual adjacencies. The issue was that networkx only attached islands using 1 nearest neighbour, so I was adding new connections to achieve 3 nearest neighbours. This has been partially solved by building a function that does 3 nearest neighbours attachments. I think an auxiliary .py file that constructed the dataframe would be the best course of action.
- Map initialisation. The seeds could be spread to cover the entire map using a systematic approach rather than taking random provinces and hoping a seed is adjacent to it. It works for a small number (~50) of provinces, it would not for the ~1000 EU NUTS-3 regions. More on this topic in the next section.
- Cooling schedule. Currently linear, but exponential is worth exploring.

These are changes that would expand the scope of the project:

- 
