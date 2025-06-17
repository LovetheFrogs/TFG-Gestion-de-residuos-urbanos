import datetime
import os
import sys
import json
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)
import time
import shutil
from ..utils import create_models as cm
from ..utils import plotter
from ..utils import utils
from ..problem.algorithms import Algorithms
from ..problem.model import Graph


class Benchmark():
    def __init__(self):
        self.MAX_CAPACITY = 15000
        self.NODES = 350
        self.CWD = os.getcwd()
        self.PATH = os.path.join(self.CWD, "Algorithm", "Library", "benchmark")
        self.data = {
            "time_divide": ([], []),
            "time_solve": ([], []),
            "time_total": ([], []),
            "value": ([], []),
            "max_nodes": ([], []),
            "avg_nodes": ([], []),
            "min_nodes": ([], []),
            "zones":([], [])
        }
        self.g = Graph()
    
    def run(self):
        print("Creating files...")
        self.create_files()
        self.g.populate_from_file(f"{self.CWD}/Algorithm/Library/benchmark/files/file.txt")
        text = "Running benchmark"
        count = 0
        utils.printProgressBar(0,
                               int((((self.MAX_CAPACITY - 1000) / 100) + 1) * 2),
                               f"{text + ' ' * (35 - len(text))}",
                               f"{count}/{int((((self.MAX_CAPACITY - 1000) / 100) + 1) * 2)}{' ' * 10}",
                               show_eta=True)
        for capacity in range(1000, self.MAX_CAPACITY + 1, 100):
            text = f"Running benchmark ({capacity} capacity)"
            count += 1
            self.run_divide_solve(capacity, False)
            utils.printProgressBar(count,
                                   int((((self.MAX_CAPACITY - 1000) / 100) + 1) * 2),
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{count}/{int((((self.MAX_CAPACITY - 1000) / 100) + 1) * 2)}{' ' * 10}",
                                   show_eta=True)
            self.run_divide_solve(capacity, True)
            count += 1
            utils.printProgressBar(count,
                                   int((((self.MAX_CAPACITY - 1000) / 100) + 1) * 2),
                                   f"{text + ' ' * (35 - len(text))}",
                                   f"{count}/{int((((self.MAX_CAPACITY - 1000) / 100) + 1) * 2)}{' ' * 10}",
                                   show_eta=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"{timestamp}_zonesGsize"
        self.RESULTS = f"{self.PATH}/results/{folder}"
        os.makedirs(f"{self.RESULTS}")
        print("Creating graphs...")
        self.create_graphs()
        print("Saving raw data...")
        self.save_results()
        self.cleanup()
    
    def create_files(self):
        cm.DATA_SIZE = 1
        cm.MAX_NODES = self.NODES
        cm.MIN_NODES = self.NODES
        cm.create_dataset()
        shutil.move(f"{self.CWD}/Algorithm/Library/utils/datasets/dataset1.txt",
                    f"{self.PATH}/files/file.txt")

        shutil.move(f"{self.CWD}/Algorithm/Library/utils/datasets/log.txt",
                    f"{self.PATH}/files/log.txt")

    def run_divide_solve(self, capacity, asc):
        res_list = int(asc)
        algo = Algorithms(self.g)
        start = time.time()
        subgraphs, z = algo.divide(zone_weight=capacity, dir="False", asc=asc)
        end = time.time()
        time_divide = (end - start) * 1000
        self.data["time_divide"][res_list].append(time_divide)
        tot_runtime = 0
        tot_value = 0
        for sg in subgraphs:
            algo = Algorithms(sg)
            start2 = time.time()
            _, v = algo.run(dir="False")
            end2 = time.time()
            tot_runtime += (end2 - start2)
            if tot_runtime < 0:
                print(f"Start: {start2}, End: {end2}, Runtime: {tot_runtime}")
            tot_value += v
        self.data["time_solve"][res_list].append(tot_runtime)
        self.data["time_total"][res_list].append(tot_runtime + (time_divide * 1000))
        self.data["value"][res_list].append(tot_value)
        self.data["zones"][res_list].append(len(z))
        
        node_counts = [len(sg) for sg in subgraphs]
        self.data["min_nodes"][res_list].append(min(node_counts))
        self.data["max_nodes"][res_list].append(max(node_counts))
        self.data["avg_nodes"][res_list].append(int(sum(node_counts) / len(node_counts)))

    def create_graphs(self):
        pltr = plotter.Plotter()
        plt_obj = pltr.plot_bench_results(self.data["time_divide"],
                                          list(range(1000, 1101, 100)),
                                          "Time to divide a graph vs. number of nodes",
                                          "Capacity",
                                          "Time",
                                          ["Ascendent division", "Descendent division"],
                                          range(1000, self.MAX_CAPACITY + 1, 100))
        plt_obj.savefig(f"{self.RESULTS}/divide_time.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_bench_results(self.data["time_solve"],
                                          list(range(1000, 1101, 100)),
                                          "Time to solve a graph vs. capacity",
                                          "Capacity",
                                          "Time",
                                          ["Ascendent division", "Descendent division"],
                                          range(1000, self.MAX_CAPACITY + 1, 100))
        plt_obj.savefig(f"{self.RESULTS}/time_solve.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_bench_results(self.data["time_total"],
                                          list(range(1000, 1101, 100)),
                                          "Total time (divide + solve) vs. capacity",
                                          "Capacity",
                                          "Time",
                                          ["Ascendent division", "Descendent division"],
                                          range(1000, self.MAX_CAPACITY + 1, 100))
        plt_obj.savefig(f"{self.RESULTS}/time_total.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_bench_results(self.data["value"],
                                          list(range(1000, 1101, 100)),
                                          "Solution value vs. capacity",
                                          "Capacity",
                                          "Value",
                                          ["Ascendent division", "Descendent division"],
                                          range(1000, self.MAX_CAPACITY + 1, 100))
        plt_obj.savefig(f"{self.RESULTS}/value.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_distribution_bar_chart(data=self.data["max_nodes"],
                                                title="Max nodes (DescDivi) & Max nodes (AscDivi)",
                                                x_label="Nodes",
                                                y_label="Occurrences",
                                                labels=["Max. nodes (DescDivi)", "Max. nodes (AscDivi)"])
        plt_obj.savefig(f"{self.RESULTS}/max_nodes.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_distribution_bar_chart(self.data["avg_nodes"],
                                                title="Avg nodes (DescDivi) & Avg nodes (AscDivi)",
                                                x_label="Nodes",
                                                y_label="Occurrences",
                                                labels=["Avg. nodes (DescDivi)", "Avg. nodes (AscDivi)"])
        plt_obj.savefig(f"{self.RESULTS}/avg_nodes.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_distribution_bar_chart(self.data["min_nodes"],
                                                title="Min nodes (DescDivi) & Min nodes (AscDivi)",
                                                x_label="Nodes",
                                                y_label="Occurrences",
                                                labels=["Min. nodes (DescDivi)", "Min. nodes (AscDivi)"])
        plt_obj.savefig(f"{self.RESULTS}/min_nodes.png")
        plt_obj.close()
        
        plt_obj = pltr.plot_bench_results(self.data["zones"],
                                          list(range(1000, 1101, 100)),
                                          "Number of zones vs. capacity",
                                          "Capacity",
                                          "Nº of zones",
                                          ["Ascendent division", "Descendent division"],
                                          range(1000, self.MAX_CAPACITY + 1, 100))
        plt_obj.savefig(f"{self.RESULTS}/zones.png")
        plt_obj.close()
    
    def save_results(self):
        with open(f"{self.RESULTS}/results.json", "w") as f:
            json.dump(self.data, f)
    
    def cleanup(self):
        os.remove(f"{self.PATH}/files/file.txt")
        os.remove(f"{self.PATH}/files/log.txt")
        

def main():
    """Runs the benchmark."""
    bench = Benchmark()
    bench.run()


if __name__ == '__main__':
    """Entry point of the script."""
    main()
    