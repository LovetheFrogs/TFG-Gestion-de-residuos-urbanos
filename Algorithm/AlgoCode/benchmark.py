"""Benchmark for different solutions for the presented problem."""

import os
import time
import shutil
from model import Graph
import create_models

class Benchmark():
    """Generates a benchmark of each of the aviable algorithms."""
    def __init__(self):
        self.avg_bfs_time = float('inf')
        self.avg_ga_tsp_time = float('inf')
        self.avg_bfs_value = float('inf')
        self.avg_ga_tsp_value = float('inf')
    
    def run(self, n: int):
        """Runs the benchmark.

        Args:
            n: Number of test graphs.
        """
        print("-#- Starting benchmark -#-")
        print(self.benchmark_bfs(n))
        print(self.benchmark_ga_tsp(n))
        print("-#- Benchmark completed -#-")
        
    def printProgressBar(
        self,
        iteration: int, 
        total: int, 
        prefix: str = '', 
        suffix: str = '', 
        decimals: int = 1,
        length: int = 100,
        fill: str = '█',
        printEnd: str = "\r"
    ):
        """Call in a loop to create terminal progress bar
        
        Thanks to StackOverflow's Greenstick user for this function, extracted
        from an answer to a thread about progressbars in python.

        Args:
            iteration: Current iteration.
            total: Total iterations.
            prefix (optional): Prefix string. Defaults to ''.
            suffix (optional): Suffix string. Defaults to ''.
            decimals (optional): positive number of decimals in percent 
                complete. Defaults to 1.
            length (optional): character length of bar. Defaults to 100.
            fill (optional): bar fill character. Defaults to '█'.
            printEnd (optional): end character (e.g. "\r", "\r\n").
                Defaults to "\r".
        """
        percent = (
            "{0:." + str(decimals) + "f}"
        ).format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()


    def benchmark_bfs(self, n: int) -> str:
        """Benchmarks the BFS function

        Args:
            n: Number of test graphs.

        Returns:
            A string signaling the end of the benchmark.
        """
        print("Benchmarking BFS...")
        cummulative_time = 0
        cummulative_value = 0
        self.printProgressBar(
            0, n, prefix="Progress:", suffix="Complete", length=50
        )
        for i in range(n):
            g = Graph()
            g.populate_from_file(
                f"{os.getcwd()}/files/datasets/dataset{i + 1}.txt"
            )
            start = time.time()
            path = g.bfs(g.center)
            end = time.time()
            cummulative_time += (end - start)
            nodes = [node.index for node in g.node_list]
            g.convert = {i: node for i, node in enumerate(nodes)}
            cummulative_value += g.evaluate(
                [n - 1 for n in path[1:]]
                                            )[0]
            self.printProgressBar(
                i + 1, n, prefix="Progress:", suffix="Complete", length=50
            )
        
        self.avg_bfs_time = (cummulative_time / n)
        self.avg_bfs_value = (cummulative_value / n)
        
        return "BFS benchmark completed ☺"
    
    def benchmark_ga_tsp(self, n: int) -> str:
        """Benchmarks the Genetic Algorithm for the TSP problem.

        Args:
            n: Number of test graphs.

        Returns:
            A string signaling the end of the benchmark.
        """
        print("Benchmarking Genetic Algorithm (TSP)...")
        cummulative_time = 0
        cummulative_value = 0
        self.printProgressBar(
            0, n, prefix="Progress:", suffix="Complete", length=50
        )
        for i in range(n):
            g = Graph()
            g.populate_from_file(
                f"{os.getcwd()}/files/datasets/dataset{i + 1}.txt"
            )
            start = time.time()
            _, value = g.run_ga_tsp(dir=f"{os.getcwd()}/files/plots", vrb=False)
            end = time.time()
            os.remove(f"{os.getcwd()}/files/plots/Path0.png")
            os.remove(f"{os.getcwd()}/files/plots/Evolution0.png")

            cummulative_time += (end- start)
            cummulative_value += value
            self.printProgressBar(
                i + 1, n, prefix="Progress:", suffix="Complete", length=50
            )

        self.avg_ga_tsp_time = (cummulative_time / n)
        self.avg_ga_tsp_value = (cummulative_value / n)
        shutil.rmtree(f"{os.getcwd()}/files/plots")

        return "Genetic Algorithm (TSP) benchmark completed ☺"

    def __repr__(self) -> str:
        """Changes the default representation of the benchmark.

        Returns:
            A string representing the benchmark.
        """
        NEW_LINE = "\n"
        msg = (
            f"The benckmark results are:{NEW_LINE}"
            f"---------------------------------"
            f"-----------------------{NEW_LINE}"
            f"      BFS algorithm:{NEW_LINE}"
            f"        |{NEW_LINE}"
            f"        |- Average time to execute: {self.avg_bfs_time}"
            f"{NEW_LINE}"
            f"        |- Average fitness value: {self.avg_bfs_value}"
            f"{NEW_LINE}{NEW_LINE}"
            f"      Genetic algorithm:{NEW_LINE}"
            f"        |{NEW_LINE}"
            f"        |- Average time to execute: {self.avg_ga_tsp_time}"
            f"{NEW_LINE}"
            f"        |- Average fitness value: {self.avg_ga_tsp_value}"
            f"{NEW_LINE}{NEW_LINE}"
        )
        return msg

def main():
    """Generates sample files and benchmarks the algorithms."""
    create_models.DATA_SIZE = 10
    create_models.MAX_NODES = 300
    create_models.MIN_NODES = 200
    print("Creating datasets...")
    create_models.create_dataset()
    
    bm = Benchmark()
    bm.run(create_models.DATA_SIZE)
    shutil.rmtree(f"{os.getcwd()}/files/datasets")
    print(bm)

if __name__ == '__main__':
    """Entry point of the script."""
    main()
