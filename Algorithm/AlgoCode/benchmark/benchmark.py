"""Benchmark for different solutions for the presented problem."""

import datetime
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
import time
import shutil
from utils import create_models as cm
from problem.algorithms import Algorithms
from problem.model import Graph
from tabulate import tabulate


BENCHMARK_SIZE = 100
MIN_FILE_SIZE = 50
MAX_FILE_SIZE = 100
CWD = os.getcwd()
VERBOSE = False
NEW_LINE = "\n"


class Benchmark():
    """Generates a benchmark of each of the aviable algorithms."""

    def __init__(self):
        self.results = {"NN": [],
                        "2opt": [],
                        "SA": [],
                        "TS": [],
                        "NN2opt": [],
                        "NNSA": [],
                        "NNTS": []}
        
    def update_global(self):
        """ Takes script call arguments (if any) and updates the value of the
        extraction constants (nº of nodes, nº of files...)
        """
        for flag, value in zip(sys.argv[1::2], sys.argv[2::2]):
            match flag:
                case "-n":
                    global BENCHMARK_SIZE
                    BENCHMARK_SIZE = int(value)
                case "-i":
                    global MIN_FILE_SIZE
                    MIN_FILE_SIZE = int(value)
                case "-j":
                    global MAX_FILE_SIZE
                    MAX_FILE_SIZE = int(value)
                case "-v":
                    global VERBOSE
                    if value == "True" or value == "1":
                        VERBOSE = True
                    else:
                        VERBOSE = False
                case _:
                    continue

    def run(self):
        """Runs the benchmark."""
        self.update_global()
        print("Creating files")
        self.create_benchmark_files()
        print("-#- Starting benchmark -#-")
        for i in range(BENCHMARK_SIZE):
            self.graph = Graph()
            if VERBOSE: print(f"Loading file {i + 1}")
            self.graph.populate_from_file(f"{CWD}/benchmark/files/file{i + 1}.txt",
                                          verbose=VERBOSE)
            self.algo = Algorithms(self.graph)
            if VERBOSE: print(f"    Running simple benchmarks")
            self.run_simple_benchmark(i)
            if VERBOSE: print(f"    Running combined benchmarks")
            self.run_combined_benchmark(i)
        self.generate_results()
        self.save_results()
        
        for i in range(BENCHMARK_SIZE):
            os.remove(f"{CWD}/benchmark/files/file{i + 1}.txt")
            
        os.remove(f"{CWD}/benchmark/files/log.txt")
        
        print("-#- Benchmark completed -#-")

    def create_benchmark_files(self):
        cm.DATA_SIZE = BENCHMARK_SIZE
        cm.MAX_NODES = MAX_FILE_SIZE
        cm.MIN_NODES = MIN_FILE_SIZE
        cm.VERBOSE = VERBOSE
        cm.create_dataset()
        
        for i in range(BENCHMARK_SIZE):
            shutil.move(f"{CWD}/utils/datasets/dataset{i + 1}.txt", 
                        f"{CWD}/benchmark/files/file{i + 1}.txt")
            
        shutil.move(f"{CWD}/utils/datasets/log.txt",
                    f"{CWD}/benchmark/files/log.txt")

    def run_simple_benchmark(self, i: int):
        _, lb = self.algo.held_karp_lb()
        print(f"    Calculating file {i + 1}")
        print(f"        Running NN for file {i + 1}")
        start = time.time()
        _, v = self.algo.nearest_neighbor(dir=f"{CWD}/benchmark", name="NN")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NN"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        print(f"        Running 2opt for file {i + 1}")
        start = time.time()
        _, v = self.algo.run_two_opt(dir=f"{CWD}/benchmark", name="2opt")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["2opt"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        print(f"        Running SA for file {i + 1}")
        start = time.time()
        _, v = self.algo.run_sa(dir=f"{CWD}/benchmark", name="SA")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["SA"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        print(f"        Running TS for file {i + 1}")
        start = time.time()
        _, v = self.algo.run_tabu_search(dir=f"{CWD}/benchmark", name="TS")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["TS"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        os.remove(f"{CWD}/benchmark/NN.png")
        os.remove(f"{CWD}/benchmark/2opt.png")
        os.remove(f"{CWD}/benchmark/SA.png")
        os.remove(f"{CWD}/benchmark/TS.png")

    def run_combined_benchmark(self, i: int):
        _, lb = self.algo.held_karp_lb()
        p, v = self.algo.nearest_neighbor(dir=f"{CWD}/benchmark", name="NN")
        if v < lb: raise ValueError("Lower bound broken")
        print(f"        Running NN+2opt for file {i + 1}")
        start = time.time()
        _, v = self.algo.run_two_opt(dir=f"{CWD}/benchmark", name="2opt",
                                     path=p)
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NN2opt"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        print(f"        Running NN+SA for file {i + 1}")
        start = time.time()
        _, v = self.algo.run_sa(dir=f"{CWD}/benchmark", name="SA",
                                path=p)
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NNSA"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        print(f"        Running NN+TS for file {i + 1}")
        start = time.time()
        _, v = self.algo.run_tabu_search(dir=f"{CWD}/benchmark", name="TS",
                                         path=p)
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NNTS"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        os.remove(f"{CWD}/benchmark/NN.png")
        os.remove(f"{CWD}/benchmark/2opt.png")
        os.remove(f"{CWD}/benchmark/SA.png")
        os.remove(f"{CWD}/benchmark/TS.png")

    def generate_results(self):
        print("Generating benchmark results")
        
        table = []
        data = ""
        data += "Results" + NEW_LINE
        headers = ("Case min(value) avg(value) max(value) " +
                 "min(time) avg(time) max(time) " +
                 "min(%) avg(%) max(%)").split()
        [*keys],[*values] = zip(*self.results.items())
        for i, l in enumerate(values):
            t = list(zip(*l))

            mt, mpt, mp = max(t[0]), max(t[1]), max(t[2])
            at, apt, ap = (sum(t[0]) / len(t[0]), 
                           sum(t[1]) / len(t[1]), 
                           sum(t[2]) / len(t[2]))
            mit, mipt, mip = min(t[0]), min(t[1]), min(t[2])
            table.append([keys[i], f"{mit:.2f}", f"{at:.2f}", f"{mt:.2f}",
                          f"{mipt:.2f}", f"{apt:.2f}", f"{mpt:.2f}",
                          f"{mip:.2f}", f"{ap:.2f}", f"{mp:.2f}"])    
        data += tabulate(table, headers, tablefmt="rounded_outline")
        self.data = data
        
        print("Results generated")

    def save_results(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{CWD}/benchmark/results/benchmark_{timestamp}.txt", "w") as file:
            file.write(self.data)

    def __repr__(self) -> str:
        """Changes the default representation of the benchmark.

        Returns:
            A string representing the benchmark results.
        """
        print(self.data)
        


def main():
    """Launches the benchmark."""
    Benchmark().run()


if __name__ == '__main__':
    """Entry point of the script."""
    main()
