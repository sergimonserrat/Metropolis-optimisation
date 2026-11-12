# Graph partitioning using simulated annealing
A coding project that groups the provinces of Spain into 7 territories of equal population. Using the Metropolis algorithm, it divides a provincial map into 7 contiguous regions by minimising the deviation from the mean population. Solving this problem is important for electoral districting in some countries and might be useful in logistics and public administration.

## How to run

### Install requirements

- **Python 3.12** or higher.
- **Linux** recommended; should also work on macOS and Windows.

```bash
pip install -r Requirements.txt
```

### Run the pipeline

The pipeline has four stages: **graph generation and analysis** (`Graph_generator.py`), **partitioning** (`Annealing.py`), **Hamming distances** (`Hamming.py`) and **results analysis** (`Flip_data.py`). Each script accepts `--graph real` for the case study or `--graph synthetic` for synthetic planar graphs.

Please note that the number of nodes is hardcoded to 51. To change it, you must edit the files.

#### Real case study (Spain provinces, 51 nodes)

```bash
# Graph analysis
python3.12 Graph_generator.py --graph real

# Simulated annealing
python3.12 Annealing.py --graph real --seeds 100 --flips 2 3

# Hamming distances between solutions
python3.12 Hamming.py --graph real

# Analysis
python3.12 Flip_data.py --graph real
```

#### Synthetic planar graphs (Voronoi graphs, 51 nodes)

```bash
# Generate synthetic graphs and analyse them
python3.12 Graph_generator.py --graph synthetic --n_graphs 20

# Annealing
python3.12 Annealing.py --graph synthetic --synthetic-id 0 --seeds 100 --flips 3
python3.12 Annealing.py --graph synthetic --synthetic-id 1 --seeds 100 --flips 3

# Hamming distances
python3.12 Hamming.py --graph synthetic --folders Synth0 Synth1

# Analysis
python3.12 Flip_data.py --graph synthetic --folders Synth0 Synth1
```

### Script options

#### `Graph_generator.py`

| Option | Description | Default |
|---|---|---|
| `--graph` | `real` or `synthetic` | `synthetic` |
| `--n_graphs` | Number of synthetic graphs to generate | `20` |

#### `Annealing.py`

| Option | Description | Default |
|---|---|---|
| `--graph` | `real` or `synthetic` | `synthetic` |
| `--synthetic-id` | ID of the synthetic graph | `0` |
| `--seeds` | Number of seeds | `100` |
| `--flips` | Flips per iteration (one or more) | `2 3` |
| `--jobs` | Parallel jobs | `12` |

> `n_iter = 10000` is hardcoded in `Annealing.py`. To change it, edit the file.

#### `Hamming.py`

| Option | Description | Default |
|---|---|---|
| `--graph` | `real` or `synthetic` | `synthetic` |
| `--folders` | List of graphs to analyse | all |
| `--jobs`| Parallel jobs | `12` |

#### `Flip_data.py`

| Option | Description | Default |
|---|---|---|
| `--graph` | `real` or `synthetic` | `synthetic` |
| `--folders` | List of graphs to analyse | all |
| `--jobs`| Parallel jobs | `12` |

### Approximate runtimes

| Stage | Time |
|---|---|
| `Graph_generator.py` | a few minutes |
| `Annealing.py` (100 seeds, 5 flips values) | ~30 min |
| `Hamming.py` (100 solutions) | ~5 min |
| `Flip_data.py` | a few minutes |

Runtimes for Intel-core i5-12500H using 12 parallel jobs.

### Notes

- **`Dataframe_previ.pkl`** must be at the project root for the real case study. It contains the 51 Spanish provinces with their populations, adjacencies and geometry.
- Outputs are created in the same folder where the files are run. It is strongly recommended to place all files in the same root to avoid errors.
- Running part of the pipeline without having run the previous steps may result in errors due to files or folders not being found.

## Generated outputs

Running the pipeline creates the following folders and files.

### Graph generation and analysis (`Graph_generator.py`)

| Folder / file | Content |
|---|---|
| `Synthetic_graphs/Synthetic{synthetic_id}.pkl` | Synthetic planar graph with 51 nodes. `POPULATION` is log-normal distributed. `TERRITORY_ID` denotes belonging to a region (is generated as 0), `geometry` is the geometry, `geometry_point` is the centroid. |
| `ccdf_poblacions.png` | CCDF (complementary cumulative distribution) of the populations, log-log plot. Generated with `--graph real`. |
| `posterior_lognormal.png` | Posterior distributions of `mu` and `sigma` from the LogNormal fit (PyMC) to the populations. Generated with `--graph real`. |

### Partitioning (`Annealing.py`)

| Folder / file | Content |
|---|---|
| `Solutions/Real/{flips}flips/` | `Repartiment_seed{seed}_iter10000_{flips}flips.pkl` (map configuration) and `Totals_seed{seed}_iter10000_{flips}flips.pkl` (aggregate data per region) for the real case. |
| `SynthSolutions/Synth{synthetic_id}/{flips}flips/` | Same files, for each synthetic graph. |
| `Maps/Real/{flips}flips/` | `map_seed{seed}_iter10000_{flips}flips.png` (solution map) for the real case. |
| `SynthMaps/Synth{synthetic_id}/{flips}flips/` | Same maps, for each synthetic graph. |
| `graph.png` / `synthgraph{synthetic_id}.png` | Plot of the underlying graph (rook connectivity + islands attached via 3-NN). |

### Hamming distances (`Hamming.py`)

| Folder / file | Content |
|---|---|
| `Distances/{graph}/Hamming_{flips}flips.npy` | Symmetric distance matrix between all pairs of solutions (minimum Hamming distance over the 5040 relabellings of 7 regions). |
| `Histograms/{graph}/Hamming_{flips}flips.png` | Histogram of the upper triangle of the distance matrix. |

Where `{graph}` is `Real` or `Synth{synthetic_id}`.

### Results analysis (`Flip_data.py`)

| Folder / file | Content |
|---|---|
| `Flip_data_{graph}.pkl` | DataFrame with the energy (sum of deviations from the mean population) per (`seed`, `flips`) combination |
| `Energy_vs_flips_{graph}.png` | Mean energy vs flips per iteration, with SEM error bars. |
| `AvgHamming_vs_flips_{graph}.png` | Mean Hamming distance between solutions vs flips per iteration, with 1-std error bars. |
| `Minimum_Hamming_{graph}.png` | Histogram of Hamming distances between the minimum-energy configurations (only generated if there is > 1 minimum). |
| `Optimal_maps/{graph}/map_seed{seed}_iter10000_{flips}flips.png` | Copies of the maps corresponding to the lowest energy configuration(s). |

Where `{graph}` is `Real` or `Synth{synthetic_id}`.

### Notes

- `Minimum_Hamming_{graph}.png` is skipped if there is only one solution with the lowest energy (the real case has degeneracy because populations are integer, but synthetic populations are float).

## Repository structure:
- `Report.pdf`: report summarising the findings of this project.
- `Graph_generator.py`: generates synthetic planar graphs. With `--graph real` it analyses graph metrics, the population CCDF (to show nodes are not power-law weighted), and computes the best LogNormal parameters with Bayesian methods.
- `Annealing.py`: plants seeds and populates the map, proposes flips with contiguity checks, and saves the resulting configuration.
- `Hamming.py`: computes Hamming distances between all pairs of solutions (accounting for region relabelling) and saves distance matrices and histograms.
- `Flip_data.py`: analyses results by flips (mean energy, mean Hamming distance), and saves the optimal configuration(s) out of all the (seeds, flips) combinations.
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

## Results and conclusions
- Most solutions are separated by 20-25 flips, which supports the hypothesis of many separated local minima. In fact, the average Hamming distance is around 22-23 flips. This is further discussed in `Report.pdf`. It is an argument in favour of building a quantum version of this optimisation problem that takes advantage of tunneling, if possible.
- 3 flips per iteration gives the lowest average energy, but the difference is not statistically significant compared to 2 or 4 flips.
- Unisng the real graph case, the lowest energy found is 244039 inhabitants. This optimal energy is shared by 15 different configurations, of which 11 share the same 3 regions, whereas the rest only share 2.
- The real case study has been succesfully generalised to planar graphs with log-normal weighted nodes, according to preliminary results.

## Future improvements
In its current form the code has deficiencies for what it is already doing. These are some things this project could do better:

- **Study of scalability**: how does the problem behave with a larger number of nodes? Either synthetic graphs or the entire NUTS3 regions could be used in the study. Is the problem manageable if the number of regions is increased?
- **Rigidity of the initial state**. The initial state fixes the amount of territories to 7 therefore there is no phase transition. Changing the energy function so it encourages the formation of 7 territories at low temperature would produce richer phenomena. Implementing an updated version with phase transitions will require the definition of an order parameter, a susceptibility and a correlation length.
- **Building a quantum version**. This is an NP-hard problem. Taking advantage of quantum tunneling should outperform the classical approach. On top of this, it is possible that the system exhibits phase transitions without a classical analogue and/or non-classical correlations (such as entanglement).
- **Study of correlations**. The best configurations share 3 out of 7 regions. That is a strong correlation. Which regions appear together more often in the catalog of solutions? Can a consensus graph be built aggregating the correlations of the entire catalog of solutions in this repository?
- **Universality classes**. How does this graph, its topology and its hypothetical phase transitions fit into the general picture? Does the graph behave like a random graph or some other kind of graph? Would I find critical exponents that could be compared with known thermodynamic systems?
