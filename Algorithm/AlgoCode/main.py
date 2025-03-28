import os
from problem import model
from problem import algorithms

"""Example usage of loading a graph and creating its tour(s)."""
def run():
    cwd = os.getcwd()

    # Create a graph from the data in ~/tests/files/test3.txt
    g = model.Graph()
    g.populate_from_file(f"{cwd}/tests/files/test3.txt", verbose=True)

    # Divide the graph into zones of 1.500kg each and create the needed 
    # subgraphs.
    algo = algorithms.Algorithms(g)
    subgraphs, zones = algo.divide(1500, dir=f"{cwd}/problem/plots",
                                   name="")

    print(f"Number of zones: {len(zones)}")
    print()

    total = 0
    # For each subgraph, calculate the lower bound, create a NN tour and use it
    # to find a tour using Tabu-Search. Save it to a file called `subgraph{i}`
    points = []
    for i, sg in enumerate(subgraphs):
        algo = algorithms.Algorithms(sg)
        print(f"Subgraph {i + 1} - Weight {sg.total_weight()}")
        print("##############################################################")
        _, lb = algo.held_karp_lb()
        print(f"|   Held-Karp lower bound = {lb:.2f}")
        nnp, nnv = algo.nearest_neighbor(dir=f"{cwd}/problem/plots", name=0)
        os.remove(f"{cwd}/problem/plots/0.png")
        print(f"|   Nearest Neighbor tour value = {nnv:.2f} "
              f"(Within {abs(100 - ((100 * nnv) / lb)):.1f}% "
              f"of the lower bound)")
        tsp, tsv = algo.run_ta(path=nnp, 
                                      dir=f"{cwd}/problem/plots", 
                                      name=f"subgraph{i + 1}")
        print(f"|   Tabu search tour value = {tsv:.2f} "
              f"(Within {abs(100 - ((100 * tsv) / lb)):.1f}% "
              f"of the lower bound)")
        print("--------------------------------------------------------------")
        print()
        total += tsv
        
        # Save coordinates of the points that form the path.
        points.append([sg.get_node(n).coordinates for n in tsp])

    print(f"Total cost = {total}")
    
    # Print all paths calculated
    algo.plot_multiple_paths(points, dir=f"{cwd}/problem/plots", name="AllPaths")
    


def main():
    run()


if __name__ == '__main__':
    main()
