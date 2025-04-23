"""Benchmark for different solutions for the presented problem.

The benchmark module has several constants, read the documentation to learn
how they can be changed. They indicate the benchmark mode, the number of sample
files, and the nodes (min and max) of such files.
"""

import datetime
import os
import sys
import json
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
import time
import shutil
from utils import create_models as cm
from utils import utils
from problem.algorithms import Algorithms
from problem.model import Graph
from tabulate import tabulate


TSPLIB_SAMPLES = 7
BENCHMARK_SIZE = 100
MIN_FILE_SIZE = 40
MAX_FILE_SIZE = 120
CWD = os.getcwd()
VERBOSE = False
MODE = 0
NEW_LINE = "\n"


class Benchmark():
    """Generates a benchmark of each of the aviable algorithms."""

    def __init__(self):
        self.data = "Empty benchmark"
        
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
                case "-m":
                    global MODE
                    if value == "True" or value == "TSPLib" or value == "1":
                        MODE = 1
                    else:
                        MODE = 0
                case _:
                    continue

    def create_variables(self):
        """Creates variables needed depending on the run mode."""
        if MODE == 0:
            self.results = {"NN": [],
                            "2opt": [],
                            "SA": [],
                            "TS": [],
                            "NN2opt": [],
                            "NNSA": [],
                            "NNTS": []}
        if MODE == 1:
            self.optimal = {"berlin52": 7542,
                            "eil76": 538,
                            "bier127": 118282,
                            "eil101": 629,
                            "kroa100": 21282,
                            "ch130": 6110,
                            "pr76": 108159
                            }
            self.results = {"berlin52": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]],
                            "eil76": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]],
                            "bier127": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]],
                            "eil101": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]],
                            "kroa100": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]],
                            "ch130": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]],
                            "pr76": [["NN"], ["2opt"], ["SA"], ["TS"], 
                                         ["NN2opt"], ["NNSA"], ["NNTS"]]
                            }

    def run(self):
        """Runs the benchmark."""
        self.update_global()
        stage = 0
        total_stages = 0
        if VERBOSE: print("-#- Starting benchmark -#-")
        if MODE == 0:
            self.create_variables()
            total_stages = (BENCHMARK_SIZE * 9) + 1
            if VERBOSE:  print("Creating files")
            self.create_benchmark_files()
            if VERBOSE:  print("Files created")
            
            if VERBOSE:
                    text = "Starting benchmark"
                    utils.printProgressBar(stage, total_stages, 
                                           f"{text + ' ' * (35 - len(text))}", 
                                           f"{stage}/{total_stages}{' ' * 10}")
            
            for i in range(BENCHMARK_SIZE):
                self.graph = Graph()
                self.graph.populate_from_file(f"{CWD}/benchmark/files/file{i + 1}.txt")
                self.algo = Algorithms(self.graph)
                stage = self.run_simple_benchmark(i, stage, total_stages)
                stage = self.run_combined_benchmark(i, stage, total_stages)
            if VERBOSE:
                text = "Algorithm stage"
                utils.printProgressBar(stage + 1, total_stages, 
                                        f"{text + ' ' * (35 - len(text))}", 
                                        f"{stage}/{total_stages}{' ' * 10}")    
            
            self.generate_results()
            self.save_results()
            
            for i in range(BENCHMARK_SIZE):
                os.remove(f"{CWD}/benchmark/files/file{i + 1}.txt")
                
            os.remove(f"{CWD}/benchmark/files/log.txt")
        
        if MODE == 1:
            self.create_variables()
            total_stages = (TSPLIB_SAMPLES * 8) + 1
            if VERBOSE:
                    text = "Starting benchmark"
                    utils.printProgressBar(stage, total_stages, 
                                           f"{text + ' ' * (35 - len(text))}", 
                                           f"{stage}/{total_stages}{' ' * 10}")
            stage = self.run_tsplib_simple(stage, total_stages)
            stage = self.run_tsplib_combined(stage, total_stages)
            if VERBOSE:
                    text = "Algorithm stage"
                    utils.printProgressBar(stage + 1, total_stages, 
                                           f"{text + ' ' * (35 - len(text))}", 
                                           f"{stage}/{total_stages}{' ' * 10}")
            
            self.generate_results()
            self.save_results()
            
        if VERBOSE: print("-#- Benchmark completed -#-")

    def create_benchmark_files(self):
        """Creates the test files if running in mode 0."""
        cm.DATA_SIZE = BENCHMARK_SIZE
        cm.MAX_NODES = MAX_FILE_SIZE
        cm.MIN_NODES = MIN_FILE_SIZE
        cm.create_dataset()
        
        for i in range(BENCHMARK_SIZE):
            shutil.move(f"{CWD}/utils/datasets/dataset{i + 1}.txt", 
                        f"{CWD}/benchmark/files/file{i + 1}.txt")
            
        shutil.move(f"{CWD}/utils/datasets/log.txt",
                    f"{CWD}/benchmark/files/log.txt")

    def run_simple_benchmark(self, i: int, stage: int, total_stages: int) -> int:
        """Runs simple algorithms for mode 0.

        Args:
            i: The graph instance being evaluated.
            stage: The stage of the benchmark.
            total_stages (int): The total number of stages of the benchmark.

        Raises:
            ValueError: If the lower bound is superior to any of the results
                obtained.

        Returns:
            The current stage of the benchmark. 
        """
        if VERBOSE:
            stage += 1
            text = f"Lower bound (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        _, lb = self.algo.held_karp_lb()
        
        if VERBOSE:
            stage += 1
            text = f"Nearest Neighgbor (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        start = time.time()
        _, v = self.algo.nearest_neighbor(dir=f"{CWD}/benchmark", name="NN")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NN"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        
        if VERBOSE:
            stage += 1
            text = f"2opt (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        start = time.time()
        _, v = self.algo.run_two_opt(dir=f"{CWD}/benchmark", name="2opt")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["2opt"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        
        if VERBOSE:
            stage += 1
            text = f"Simulated Annealing (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        start = time.time()
        _, v = self.algo.run_sa(dir=f"{CWD}/benchmark", name="SA")
        
        if VERBOSE:
            stage += 1
            text = f"Tabu Search (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["SA"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
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
        
        return stage

    def run_combined_benchmark(self, i: int, stage: int, total_stages: int) -> int:
        """Runs combined algorithms for mode 0.

        Args:
            i: The graph instance being evaluated.
            stage: The stage of the benchmark.
            total_stages (int): The total number of stages of the benchmark.

        Raises:
            ValueError: If the lower bound is superior to any of the results
                obtained.

        Returns:
            The current stage of the benchmark. 
        """
        if VERBOSE:
            stage += 1
            text = f"Lower bound + NN (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        _, lb = self.algo.held_karp_lb()
        p, v = self.algo.nearest_neighbor(dir=f"{CWD}/benchmark", name="NN")
        if v < lb: raise ValueError("Lower bound broken")
        
        if VERBOSE:
            stage += 1
            text = f"NN + 2opt (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        start = time.time()
        _, v = self.algo.run_two_opt(dir=f"{CWD}/benchmark", name="2opt",
                                     path=p)
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NN2opt"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        
        if VERBOSE:
            stage += 1
            text = f"NN + SA (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
        start = time.time()
        _, v = self.algo.run_sa(dir=f"{CWD}/benchmark", name="SA",
                                path=p)
        end = time.time()
        if v < lb: raise ValueError("Lower bound broken")
        self.results["NNSA"].append([v, (end - start),
                                   abs(100 - ((100 * v) / lb))])
        
        if VERBOSE:
            stage += 1
            text = f"NN + Tabu search (graph {i + 1})"
            utils.printProgressBar(stage, total_stages,
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{stage}/{total_stages}{' ' * 10}")
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
        
        return stage

    def run_tsplib_simple(self, stage: int, total_stages: int) -> int:
        """Runs simple algorithms for mode 1.

        Args:
            stage: The stage of the benchmark.
            total_stages (int): The total number of stages of the benchmark.

        Raises:
            ValueError: If the lower bound is superior to any of the results
                obtained.

        Returns:
            The current stage of the benchmark. 
        """
        for test, opt in self.optimal.items():
            g = Graph()
            g.populate_from_tsplib(f"{CWD}/benchmark/TSPLib/{test}.tsp")
            algo = Algorithms(g)
            if VERBOSE:
                stage += 1
                text = f"Nearest Neighgbor ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            
            start = time.time()
            _, v = algo.nearest_neighbor(dir=f"{CWD}/benchmark", name="NN")
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][0].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
            if VERBOSE:
                stage += 1
                text = f"2opt ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            start = time.time()
            _, v = algo.run_two_opt(dir=f"{CWD}/benchmark", name="2opt")
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][1].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
            if VERBOSE:
                stage += 1
                text = f"Simulated Annealing ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            start = time.time()
            _, v = algo.run_sa(dir=f"{CWD}/benchmark", name="SA")
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][2].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
            if VERBOSE:
                stage += 1
                text = f"Tabu Search ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            start = time.time()
            _, v = algo.run_tabu_search(dir=f"{CWD}/benchmark", name="TS")
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][3].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
        os.remove(f"{CWD}/benchmark/NN.png")
        os.remove(f"{CWD}/benchmark/2opt.png")
        os.remove(f"{CWD}/benchmark/SA.png")
        os.remove(f"{CWD}/benchmark/TS.png")
        
        return stage

    def run_tsplib_combined(self, stage: int, total_stages: int) -> int:
        """Runs combined algorithms for mode 1.

        Args:
            stage: The stage of the benchmark.
            total_stages (int): The total number of stages of the benchmark.

        Raises:
            ValueError: If the lower bound is superior to any of the results
                obtained.

        Returns:
            The current stage of the benchmark. 
        """
        for test, opt in self.optimal.items():
            g = Graph()
            g.populate_from_tsplib(f"{CWD}/benchmark/TSPLib/{test}.tsp")
            algo = Algorithms(g)
            if VERBOSE:
                stage += 1
                text = f"NN 2nd stage ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            
            p, v = algo.nearest_neighbor(dir=f"{CWD}/benchmark", name="NN")
            if v < opt: raise ValueError("Optimal value broken")
            
            if VERBOSE:
                stage += 1
                text = f"NN + 2opt ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            start = time.time()
            _, v = algo.run_two_opt(dir=f"{CWD}/benchmark", name="2opt",
                                     path=p)
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][4].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
            if VERBOSE:
                stage += 1
                text = f"NN + SA ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            start = time.time()
            _, v = algo.run_sa(dir=f"{CWD}/benchmark", name="SA",
                                     path=p)
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][5].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
            if VERBOSE:
                stage += 1
                text = f"NN + Tabu search ({test})"
                utils.printProgressBar(stage, total_stages,
                                    f"{text + ' ' * (35 - len(text))}",
                                    f"{stage}/{total_stages}{' ' * 10}")
            start = time.time()
            _, v = algo.run_tabu_search(dir=f"{CWD}/benchmark", name="TS",
                                     path=p)
            end = time.time()
            if v < opt: raise ValueError("Optimal value broken")
            self.results[test][6].append([v, (end - start),
                                   abs(100 - ((100 * v) / opt))])
            
        os.remove(f"{CWD}/benchmark/NN.png")
        os.remove(f"{CWD}/benchmark/2opt.png")
        os.remove(f"{CWD}/benchmark/SA.png")
        os.remove(f"{CWD}/benchmark/TS.png")
            
        return stage
            
    def generate_results(self):
        """Generates the result sheet."""
        if VERBOSE: print("Generating results")
        
        table = []
        data = ""
        if MODE == 0:
            data += f"Results - {BENCHMARK_SIZE} file(s), between {MIN_FILE_SIZE} "
            data += f"and {MAX_FILE_SIZE} nodes." + NEW_LINE
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

        if MODE == 1:
            data += f"Results - TSPLib" + NEW_LINE
            headers = ("Instance Optimal Method Value Time %").split()
            for k, v in self.results.items():
                for l in v:
                    if l[0] == "NN":
                        table.append([k, self.optimal[k], l[0], 
                                      f"{l[1][0]:.2f}", f"{l[1][1]:.2f}",
                                      f"{l[1][2]:.2f}"])
                    else:
                        table.append(["-", self.optimal[k], l[0], 
                                      f"{l[1][0]:.2f}", f"{l[1][1]:.2f}", 
                                      f"{l[1][2]:.2f}"])
            
            data += tabulate(table, headers, tablefmt="rounded_outline")
               
        self.data = data   
        if VERBOSE: print("Results generated")

    def save_results(self):
        """Dumps results to a text file and raw data to a JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"{CWD}/benchmark/results/benchmark_{timestamp}"
        os.mkdir(f"{path}")
        with open(f"{path}/results.txt", "w") as file:
            file.write(self.data)
        with open(f"{path}/data.json", "w") as file:
            json.dump(self.results, file)

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
