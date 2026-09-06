# Graph partitioning using Metropolis optimisation
A coding project that groups the provinces of Spain into 7 territories of equal population. Using the Metropolis algorithm, it divides a provincial map into 7 contiguous regions by minimising the deviation from the mean population. Solving this problem is important for electoral districting in some countries and might be useful in logistics and public administration.

## Repository structure:
- `Projecte geopandas v3.py`: Main code
- `Dataframe_previ.pkl`: Graph and population data. 

| Column | Type | Description |
|---|---|---|
| NUTS_ID | string | EU NUTS-3 Identifier |
| NUTS_NAME | string | Province name |
| geometry | multipolygon | Geometry of the province |
| CENTROID | point | Coordinates of the centre of the province |
| TERRITORY_ID | integer | ID number (1-7) of the territory the province belongs to. Initialises at 0. |
| POPULATION | integer | Population of the province |
| VESINS | list | Neighbouring provinces. Adjacency list of the graph |

- `Requirements.txt`: list of necessary libraries and their versions
- `README.md`: this file
- `Convergence.png`: example plot showing how the optimiser converges
- `Solutions`: folder containing .pkl files describing the partition of seeds 0-9.
  - `Repartiment_seed[seed_number]_iter[number_of_iterations]_exp.pkl`: Dataframes with the same structure than `Dataframe_previ.pkl` where TERRITORY_ID has been filled with integers 1-7.
  - `Totals_seed[seed_number]_iter[number_of_iterations]_exp.pkl`: Aggregate dataframe detailing the population of the 7 regions and their deviation from the mean.
  | Column | Type | Description |
  |---|---|---|
  | TERRITORY_ID | integer | ID number (1-7) of the region the province belongs to. |
  | Membres | integer | Number of provinces with the same TERRITORY_ID |
  | Poblacio | integer | Population of the region |
  | Desviacio | float | Deviation from the mean |
- `Maps`: folder containing .png files with the solution maps of seeds 0-99. They are stored with the naming convention `Map_seed[seed_number]_iter[number_of_iterations]_exp.pkl`.
- `Graph.png`: plot showing how the provinces are connected
- `Hamming.py`: auxiliary code that computes the Hamming distance between solutions.

## Code description 
`Projecte_geopandas_v3.py`:
- Create a graph with rook connectivity using libpysal.
- Attach islands using 3 nearest neighbours.
- Plant seeds for each territory and expand them until they occupy the map. 
- Flips are proposed. Connectivity is checked to ensure the regions are contiguous. Metropolis algorithm is implemented with an exponential temperature decrease schedule.
- Creates the folders `Maps`, `Solutions` and `Convergence` if they do not already exist.
- Best map configuration is saved as `Repartiment_seed[seed_number]_iter[number_of_iterations]_exp.pkl` and the aggregate data for each TERRITORY_ID is stored in `Totals_seed[seed_number]_iter[number_of_iterations]_exp.pkl`. Both files are stored in the `Solutions` folder.
- Create 3 plots for a given seed: one for the graph `graph.png`; one for the map configuration `Map_seed[seed_number]_iter[number_of_iterations]_exp.pkl`, stored in the folder `Maps`, and one for the convergence `Convergence_seed[seed_number]_iter[number_of_iterations]_exp.pkl`, stored in the folder `Convergence`.

`Hamming.py`: 
- Computes the Hamming distance between all pairs of solutions stored in the folder `Solutions`.
- Builds a distance matrix and a histogram `Hamming_100_seeds.png` of the distances.
- Performs a non-metric (`metric_mds = False`) 3D MDS `sol_landscape_100_seeds.png` to visualise the space of solutions.

## Results and conclusions
- Convergence is fast for the first 2000-3000 iterations, then stalls (diminishing returns).
- Different runs yield different map configurations that appear qualitatively distinct. Hamming distance has been chosen to quantitatively measure distinctiveness.
- A "good" solution is easy to achieve but the optimal one is difficult to find (abundance of local minima). This makes me think that the space of solutions might look either like "gravel pavement" (flat at a large scale but full of shallow local minima) or "karstic" (deep wells separated by almost vertical walls. A spiky landscape).
- Most solutions are separated by 15-25 flips, which supports the hypothesis of many separated local minima. This is shown by the histogram `Hamming_100_seeds.png` and the MDS `sol_landscape_100_seeds.png`, where we see a distribution centered around Hamming distance ~20 in the former and a (visually) non-clustered landscape in the latter. Both are an argument in favour of building a quantum version of this optimisation problem that takes advantage of tunneling, if possible.

## Future improvements
In its current form the code has deficiencies for what it is already doing. These are some things this project could do better:

- **Build the graph and dataframe in situ**. Right now, the code takes a prebuilt dataframe that I created by manually adding a few adjacencies to a graph that was automatically built with networkx. My intention to scale up to all EU NUTS-3 regions would require adding many more manual adjacencies. The issue was that networkx only attached islands using 1 nearest neighbour, so I was adding new connections to achieve 3 nearest neighbours. This has been partially solved by building a function that does 3 nearest neighbours attachments. I think an auxiliary .py file that constructed the dataframe would be the best course of action.

These are changes that would expand the scope of the project:

- **Rigidity of the initial state**. The initial state fixes the amount of territories to 7 therefore there is no phase transition. Changing the energy function so it encourages the formation of 7 territories at low temperature would produce richer phenomena. Implementing an updated version with phase transitions will require the definition of an order parameter, a susceptibility and a correlation length.
- **Building a quantum version**. This appears to be an NP or NP-hard problem, taking advantage of quantum tunneling should outperform the classical approach. On top of this, it is possible that the system exhibits phase transitions without a classical analogue and/or non-classical correlations (such as entanglement).
- **Study of correlations**. Using 1 nearest neighbours to attach the islands introduced strong correlations between the island and the region it was attached to, difficulting the exploration of map configuration space. 3 nearest neighbours softens this issue. 
- **Universality classes**. How does this graph, its topology and its hypothetical phase transitions fit into the general picture? Does the graph behave like a random graph or some other kind of graph? Would I find critical exponents that could be compared with known thermodynamic systems?
