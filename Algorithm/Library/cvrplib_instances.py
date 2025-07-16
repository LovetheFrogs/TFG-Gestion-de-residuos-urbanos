import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from .problem import model
from .problem import algorithms
"""Example usage of loading a graph and creating its tour(s)."""


def run(name, cap):
    cwd = os.getcwd()
    inst_name = name
    capacity = cap
    os.makedirs(f"{cwd}/Algorithm/Library/CVRPlib_res/{inst_name}", exist_ok=True)

    # Create a graph from the data in ~/tests/files/test3.txt
    g = model.Graph()
    g.populate_from_cvrplib(f"{cwd}/Algorithm/Library/tests/files/CVRPLIB/{inst_name}.txt")

    # Divide the graph into zones of 1.500kg each and create the needed
    # subgraphs.
    algo = algorithms.Algorithms(g)
    subgraphs, zones = algo.divide(capacity, dir=f"{cwd}/Algorithm/Library/CVRPlib_res/{inst_name}", name=f"{inst_name}")

    total = 0
    total_time = 0
    # For each subgraph, calculate the lower bound, create a NN tour and use it
    # to find a tour using the best algorithm. Save it to a file called 
    # `subgraph{i}.txt`.
    points = []
    for i, sg in enumerate(subgraphs):
        algo = algorithms.Algorithms(sg)
        if (sg.total_weight() > capacity):
            raise ValueError
        start = time.time()
        rp, rv = algo.run(dir=f"{cwd}/Algorithm/Library/CVRPlib_res/{inst_name}",
                            name=f"aux")
        end = time.time()
        total += rv
        total_time += (end - start)

        # Save coordinates of the points that form the path.
        points.append([sg.get_node(n).coordinates for n in rp])

    print(f"Map {inst_name}")
    print(f"Zones = {len(zones)}")
    print(f"Total cost = {total:.2f}")
    print(f"Total time = {total_time:.2f}")
    print()

    # Print all paths calculated
    algo.plot_multiple_paths(points,
                             dir=f"{cwd}/Algorithm/Library/CVRPlib_res/{inst_name}",
                             name="AllPaths")


def main():
    run("X-n115-k10", 169)
    run("X-n157-k13", 12)
    run("X-n209-k16", 101)
    run("X-n256-k16", 1225)
    run("X-n393-k38", 78)
    run("X-n856-k95", 9)


if __name__ == '__main__':
    main()
