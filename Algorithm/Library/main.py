import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from .problem import model
from .problem import algorithms
"""Example usage of loading a graph and creating its tour(s)."""


def run():
    cwd = os.getcwd()

    # Create a graph from the data in ~/tests/files/test3.txt
    g = model.Graph()
    g.populate_from_file(f"{cwd}/Algorithm/Library/tests/files/test3.txt", verbose=True)

    # Divide the graph into zones of 1.500kg each and create the needed
    # subgraphs.
    algo = algorithms.Algorithms(g)
    subgraphs, zones = algo.divide(1500, dir=f"{cwd}/Algorithm/Library/problem/plots", name="")

    print(f"Number of zones: {len(zones)}")
    print()

    total = 0
    # For each subgraph, calculate the lower bound, create a NN tour and use it
    # to find a tour using the best algorithm. Save it to a file called 
    # `subgraph{i}.txt`.
    points = []
    for i, sg in enumerate(subgraphs):
        algo = algorithms.Algorithms(sg)
        print(f"Subgraph {i + 1} - Weight {sg.total_weight()} - Nodes {sg.nodes}")
        print("##############################################################")
        start = time.time()
        rp, rv = algo.run(dir=f"{cwd}/Algorithm/Library/problem/plots",
                            name=f"subgraph{i + 1}")
        end = time.time()
        print(f"|   Final tour value = {rv:.2f} ")
        print(f"|   Time to compute = {end - start}")
        print("--------------------------------------------------------------")
        print()
        total += rv

        # Save coordinates of the points that form the path.
        points.append([sg.get_node(n).coordinates for n in rp])

    print(f"Total cost = {total}")

    # Print all paths calculated
    algo.plot_multiple_paths(points,
                             dir=f"{cwd}/Algorithm/Library/problem/plots",
                             name="AllPaths")


def main():
    run()


if __name__ == '__main__':
    main()
