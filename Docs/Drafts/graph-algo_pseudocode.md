# Graph algorithm and route search pseudocode

After analising multiple options for the algorithm, the final contenders have been the [**k-means clustering**](https://en.wikipedia.org/wiki/K-means_clustering) algorithm and the [**Markov Cluster algorithm (MCL)**](https://sites.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf). Considering the map (graph) would not always have a clear clustering, and that it may be better to separate the clusters on more than the heuristics defined by the MCL algo, the k-means clustering algorithm has been decided upon.

## K-means clustering

This algorithm picks *k* random points (means) in the graph, then, for each mean, a cluster is created and a new mean is obtained using the centroid of each cluster. This is repeated until convergence has been reached.

The method of finding the new centroid (objective function) is the WCSS. Changing it may make convergence imposible.

The pseudocode for this algorithm is:

```python
def k_means_cluster(k, nodes):
    # Initialization: choose k centroids (Forgy, Random Partition, etc.)
    centroids = [c1, c2, ..., ck]
    
    # Initialize clusters list
    clusters = [[] for _ in range(k)]
    
    # Loop until convergence
    converged = false
    while not converged:
        # Clear previous clusters
        clusters = [[] for _ in range(k)]
    
        # Assign each point to the "closest" centroid 
        for node in nodes:
            distances_to_each_centroid = [distance(node, centroid) for centroid in centroids]
            cluster_assignment = argmin(distances_to_each_centroid)
            node.distance_to_centroid = distance(node, centroid)
            clusters[cluster_assignment].append([node])
        
        # Calculate new centroids
        #   (the standard implementation uses the mean of all points in a
        #     cluster to determine the new centroid)
        new_centroids = [calculate_centroid(cluster) for cluster in clusters]
        
        converged = (new_centroids == centroids)
        centroids = new_centroids
        
        if converged:
            return clusters
```

## Improving upon the first drafted solution

One of the first improvements that comes to mind is changing the objective function. In this pseudocode, this funtion is `distance(node, centroid)`. As mantioned above, changing this may result on the algorithm not converging. Two steps help in avoiding it ends up stuck in an infinite loop.

### Selecting an arbitrary maximum number of iterations i.
This may make the solution stride further from the optimal, but ensures there will be a solution regardless. To include this, the pseudocode would be changed by adding a check for i in the while loop:

`while (not converged) or (iterations >= i)`

### Making sure the distance is still the most important value of the objective function

Changing the `distance` function while keeping the Euclidean distance the most important factor of it will make convergence easier to obtain. Curently, it is calculated as:

$$\\Distance(node, centroid) = \sqrt{(node.x - centroid.x)^2 + (node.y - centroid.y)^2}$$

A new value can be obtained with the following proposed formula:

$$\ f(n, c) = Distance(n, c) * x + Weight(n) * y$$

Where:

$$\ f = Objective function $$
$$\ n = Node $$
$$\ c = Centroid $$
$$\ Distance(n, c) = Euclidean\ distance $$
$$\ Weight(n) = Estimated\ weight\ of\ the\ node $$
$$\ x, y \in [0, 1] = Weight\ of\ each\ value $$

With this function, heavier nodes will have a greater value, making it more likely that they get added to a cluster with a closer centroid.

## Further improvement

Once the $k$ clusters are created, the following function will be ran to make sure all of them have similar total weights:

```python
def get_cluster_weight(cluster):
    weight = 0
    for node in cluster:
        weight += node.weight
    
    return weight
```

If the weight is too diferent between clusters, The critical nodes (furthest from centroids) will be analised to check if it would be better to change them to another cluster.

## After steps

We can now consider each cluster as a diferent graph that will be assinged to a truck. Two options are:

1. **Find a Hamiltonian Cycle using a genetic algorithm/branch and bound.**
2. **Use Christofides' Algorithm (aproximation to the TSP).**

## Glosary

* **Node:** A data structure containing the weight of the container, the distance to the centroid and other relevant information.
