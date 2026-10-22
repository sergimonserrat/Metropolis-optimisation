# Graph partitioning using Metropolis optimisation
A coding project that groups the provinces of Spain into 7 territories of equal population. Using the Metropolis algorithm, it divides a provincial map into 7 contiguous regions by minimising the deviation from the mean population. Solving this problem is important for electoral districting in some countries and might be useful in logistics and public administration.

## Repository structure:
- `Report.pdf`: report summarising the findings of this project.
- `Annealing.py`: Main code
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

- `Requirements.txt`: list of necessary libraries and their versions.
- `README.md`: this file.
- `Convergence`: folder with plots showing how the optimiser converges.
- `Solutions`: folder containing .pkl files describing the partition of seeds 0-9.
  - `Repartiment_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl`: Dataframes with the same structure than `Dataframe_previ.pkl` where TERRITORY_ID has been filled with integers 1-7.
  - `Totals_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl`: Aggregate dataframes detailing the population of the 7 regions and their deviation from the mean.

  | Column | Type | Description |
  |---|---|---|
  | TERRITORY_ID | integer | ID number (1-7) of the region the province belongs to. |
  | Membres | integer | Number of provinces with the same TERRITORY_ID |
  | Poblacio | integer | Population of the region |
  | Desviacio | float | Deviation from the mean |

- `Maps`: folder containing .png files with the solution maps of seeds 0-99. They are stored with the naming convention `Map_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl`.
- `Optimal_maps`: folder with the 15 maps with the most optimal configuration. 
- `Distances`: folder with the distance matrices in `.npy` format.
- `graph.png`: plot showing how the provinces are connected
- `Hamming.py`: auxiliary code that computes the Hamming distance between solutions.
- `Histograms`: folder containing histograms of all the Hamming distances between pairs of solutions with seeds 0-99 for a given number of flips/iteration.
- `Flip_data.py`: auxiliary code used to analyse the results grouped by number of flips/iteration. 
- `Energy_vs_flips.png`: plot of the average energy for every value of `flips`.
- `Flip_data.pkl`: dataframe with the energy for each combination of (`seed`, `flips`).

| Column | Type | Description |
|---|---|---|
| Energy | float | Energy. |
| Seed | int | Seed. |
| Flips | int | Number of flips per iteration. |

- `Minimum_Hamming.png`: histogram of Hamming distances between the 15 optimal configurations.
- `regions_comunes.png`: map with the 3 regions shared by 14 out of the 15 optimal configurations.

## Code description 
`Projecte_geopandas_v3.py`:
- Number of seeds and number of flips/iteration can be chosen as parameters.
- Create a graph with rook connectivity using libpysal.
- Attach islands using 3 nearest neighbours.
- Plant seeds for each territory and expand them until they occupy the map. 
- Flips are proposed. Connectivity is checked to ensure the regions are contiguous. Metropolis algorithm is implemented with an exponential temperature decrease schedule.
- Creates the folders `Maps`, `Solutions` and `Convergence` if they do not already exist.
- Best map configuration is saved as `Repartiment_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl` and the aggregate data for each TERRITORY_ID is stored in `Totals_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl`. Both files are stored in the `Solutions` folder. Each value of `flips` generates a subfolder inside `Solutions`.
- Create 3 plots for a given seed: one for the graph `graph.png`; one for the map configuration `Map_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl`, stored in the folder `Maps`, and one for the convergence `Convergence_seed[seed_number]_iter[number_of_iterations]_[flips/iteration]flips.pkl`, stored in the folder `Convergence`.

`Hamming.py`: 
- Computes the Hamming distance between all pairs of solutions stored in the folder `Solutions`.
- Builds a distance matrix and a histogram `Hamming_100_seeds.png` of the distances.
- Performs a non-metric (`metric_mds = False`) 3D MDS `sol_landscape_[flips/iteration]flips.png` to visualise the space of solutions.
- Distance matrices are stored in the `Distances` folder.

`Flip_data.py`:
- Creates `flip_data.pkl`, a dataframe with the energy of every combination of `seed` and `flips`.
- Analyses the mean energy of each value of `flips`to find.
- Finds the (`seed`, `flips`) with the lowest energy and verifies that they are all distinct. Generates the corresponding histogram `Minimum_Hamming.png`.
- Loads the distance matrices generated by `Hamming.py` and computes the mean, standard deviation, and Q1 and Q3 quartiles.

## Results and conclusions
- Most solutions are separated by 20-25 flips, which supports the hypothesis of many separated local minima. In fact, the average Hamming distance is around 22-23 flips. This is further discussed in `Report.pdf`. It is an argument in favour of building a quantum version of this optimisation problem that takes advantage of tunneling, if possible.
- 3 flips per iteration gives the lowest average energy, but the difference is not statistically significant compared to 2 or 4 flips.
- The lowest energy found is 244039 inhabitants. This optimal energy is shared by 15 different configurations, of which 14 share the same 3 regions, whereas the remaining one only shares 2.

## Future improvements
In its current form the code has deficiencies for what it is already doing. These are some things this project could do better:

- **Characterise the graph**: Study the basic features of the graph and classifying it into one or more categories. The graph is suspected to be sparse. Attempt to generalise this case study to synthetic sparse graphs with power-law weighted nodes.
- **Study of scalability**: how does the problem behave with a larger number of nodes? Either synthetic graphs or the entire NUTS3 regions could be used in the study. Is the problem manageable is the number of regions is increased?
- **Rigidity of the initial state**. The initial state fixes the amount of territories to 7 therefore there is no phase transition. Changing the energy function so it encourages the formation of 7 territories at low temperature would produce richer phenomena. Implementing an updated version with phase transitions will require the definition of an order parameter, a susceptibility and a correlation length.
- **Building a quantum version**. This is an NP-hard problem. Taking advantage of quantum tunneling should outperform the classical approach. On top of this, it is possible that the system exhibits phase transitions without a classical analogue and/or non-classical correlations (such as entanglement).
- **Study of correlations**. The best configurations share 3 out of 7 regions. That is a strong correlation. Which regions appear together more often in the catalog of solutions? Can a consensus graph be built aggregating the correlations of the entire catalog of solutions in this repository?
- **Universality classes**. How does this graph, its topology and its hypothetical phase transitions fit into the general picture? Does the graph behave like a random graph or some other kind of graph? Would I find critical exponents that could be compared with known thermodynamic systems?
