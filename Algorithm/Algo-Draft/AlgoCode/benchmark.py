"""Benchmark for different solutions for the presented problem."""

# Check https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters for progress bar :) - give credit this guy is lit.

import os
import time
import shutil
from model import Graph
import create_models

class Benchmark():
    def __init__(self):
        self.avg_bfs_time = float('inf')
        self.avg_prim_time = float('inf')
        self.avg_ga_tsp_time = float('inf')
        self.avg_bfs_value = float('inf')
        self.avg_prim_value = float('inf')
        self.avg_ga_tsp_value = float('inf')
    
    def run(n: int):
        print("-#- Starting benchmark -#-")
        self.benchmark_bfs(n)
        self.benchmark_prim(n)
        self.benchmark_ga_tsp(n)
        print("-#- Benchmark completed -#-")
        

    def benchmark_bfs(n: int):
        print("Benchmarking BFS...")
        cummulative_time = 0
        cummulative_value = 0
        for i in range(n):
            g = Graph()
            g.populate_from_file(
                f"{os.getcwd()}/files/datasets/dataset{i}.txt"
            )
            start = time.time()
            path = g.bfs(g.center)
            end = time.time()
            cummulative_time += (end - start)
            cummulative_value += g.evaluate(path + [g.center.index])
        
        self.avg_bfs_time = (cummulative_time / n)
        self.avg_bfs_value = (cummulative_value / n)
        
        return "BFS benchmark completed ☺"

    def benchmark_prim(n: int):
        print("Benchmarking Prim...")
        cummulative_time = 0
        cummulative_value = 0

        for i in range(n):
            g = Graph()
            g.populate_from_file(
                f"{os.getcwd()}/files/datasets/dataset{i}.txt"
            )
            start = time.time()
            
            end = time.time()
            cummulative_time += (end - start)

        self.avg_prim_time = (cummulative_time / n)
        self.avg_prim_value = (cummulative_value / n)

        return "Prim benchmark completed ☺"
    
    def benchmark_ga_tsp(n: int):
        print("Benchmarking Genetic Algorithm (TSP)...")
        cummulative_time = 0
        cummulative_value = 0

        for i in range(n):
            g = Graph()
            g.populate_from_file(
                f"{os.getcwd()}/files/datasets/dataset{i}.txt"
            )
            start = time.time()
            _, value = g.run_ga_tsp(f"{os.getcwd()}/files/plots")
            end = time.time()

            cummulative_time += (end- start)
            cummulative_value += value

        self.avg_ga_tsp_time = (cummulative_time / n)
        self.avg_ga_tsp_value = (cummulative_value / n)
        shutil.rmtree(f"{os.getcwd()}/files/plots")

        return "Genetic Algorithm (TSP) benchmark completed ☺"

    def __repr__():
        pass

def main():
    """Generates sample files and benchmarks the algorithms."""
    create_models.DATA_SIZE = 100
    create_models.MAX_NODES = 300
    create_models.MIN_NODES = 200
    create_models.create_datasest()
    
    bm = Benchmark()
    bm.run(100)
    shutil.rmtree(f"{os.getcwd()}/files/datasets")
    print(bm)

if __name__ == '__main__':
    """Entry point of the script."""
    main()
