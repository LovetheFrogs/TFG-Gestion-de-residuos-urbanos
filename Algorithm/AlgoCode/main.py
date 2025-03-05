"""Runner code to use graphs and path-finders."""

import os
from model import Graph
from algorithms import Algorithms


def run():
    """An example of using the model & algorithms modules.
    
    In this example a graph is created from a file, divided in zones & 
    subgraphs and paths are calculated for each subgraph. A path is also
    calculated for the initial graph using both TSP and VSP.
    """
    g = Graph()
    print("Loading graph")
    g.populate_from_file(os.getcwd() + "/files/test2.txt")
    #g.populate_from_file(os.getcwd() + "/Algorithm/AlgoCode/files/test2.txt")
    print("Graph loaded")
    algo = Algorithms(g)
    _, v = algo.run_ga_tsp(ngen=500,
                           pop_size=500,
                           idx=0,
                           dir=os.getcwd() + "/plots",
                           vrb=False)
    print(f"Total value (TSP): {v}")
    res = g.divide_graph(725)
    print(f"Zone count (TSP): {len(res)}")
    sg = []
    for i, z in enumerate(res):
        sg.append(g.create_subgraph(z))
    t = 0
    for i, graph in enumerate(sg):
        algo2 = Algorithms(graph)
        p, v = algo2.run_ga_tsp(idx=i + 1,
                                vrb=False,
                                dir=os.getcwd() + "/plots")
        print(p)
        t += v
    print(f"Total value (TSP zoned): {t}")

    t = 0
    n = g.set_num_zones(725) + int(g.set_num_zones(725) * 0.1) + 1
    print(f"Zone count (VRP): {n}")
    p, v = algo.run_ga_vrp(3,
                           725,
                           ngen=1000,
                           idx=len(sg) + 1,
                           dir=os.getcwd() + "/plots",
                           vrb=False)
    for sp in p:
        current = g.get_node(sp[0])
        t += algo.evaluate(sp[1:-1])
    print(f"Total value (VRP): {t}")


def main():
    run()


if __name__ == '__main__':
    main()
