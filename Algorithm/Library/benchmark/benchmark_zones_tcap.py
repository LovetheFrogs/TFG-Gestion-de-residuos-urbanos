import datetime
import os
import json
import time
import shutil
import argparse
import pickle
import statistics
import multiprocessing
import zipfile
from tqdm import tqdm
from ..utils import create_models as cm
from ..utils import plotter
from ..utils import utils
from ..problem.algorithms import Algorithms
from ..problem.model import Graph

REPETITIONS = 10


def run_divide_solve_task(args):
    graph_bytes, capacity, asc = args
    g = pickle.loads(graph_bytes)
    res_list = int(asc)

    temp_metrics = {
        'time_divide': [],
        'time_solve': [],
        'time_total': [],
        'value': [],
        'zones': [],
        'min_nodes': [],
        'max_nodes': [],
        'avg_nodes': []
    }

    for _ in range(REPETITIONS):
        algo = Algorithms(g)
        start = time.time()
        subgraphs, z = algo.divide(zone_weight=capacity, dir="False", asc=asc)
        end = time.time()
        time_divide = (end - start) * 1000

        tot_runtime = 0
        tot_value = 0
        node_counts = []

        for sg in subgraphs:
            algo = Algorithms(sg)
            start2 = time.time()
            _, v = algo.run(dir="False")
            end2 = time.time()
            tot_runtime += (end2 - start2)
            tot_value += v
            node_counts.append(len(sg))

        temp_metrics['time_divide'].append(time_divide)
        temp_metrics['time_solve'].append(tot_runtime * 1000)
        temp_metrics['time_total'].append(tot_runtime * 1000 + time_divide)
        temp_metrics['value'].append(tot_value)
        temp_metrics['zones'].append(len(z))
        temp_metrics['min_nodes'].append(min(node_counts))
        temp_metrics['max_nodes'].append(max(node_counts))
        temp_metrics['avg_nodes'].append(sum(node_counts) / len(node_counts))

    return {
        'res_list': res_list,
        'capacity': capacity,
        'averaged': {
            k: int(statistics.mean(v)) if k.startswith("avg_nodes") else statistics.mean(v)
            for k, v in temp_metrics.items()
        }
    }


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

    def run(self, dist):
        print("Creating files...")
        self.create_files(dist)
        self.g.populate_from_file(f"{self.CWD}/Algorithm/Library/benchmark/files/file.txt")

        graph_bytes = pickle.dumps(self.g)

        tasks = []
        capacities = list(range(1000, self.MAX_CAPACITY + 1, 100))
        for capacity in capacities:
            for asc in [False, True]:
                tasks.append((graph_bytes, capacity, asc))

        num_cores = max(1, multiprocessing.cpu_count()//4)
        print(f"Using {num_cores} parallel workers...")

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(run_divide_solve_task, tasks), total=len(tasks), desc="Benchmarking",
                                ncols=90))

        capacity_index = {c: i for i, c in enumerate(capacities)}

        # Prepare space in advance
        for k in self.data:
            self.data[k] = ([0] * len(capacities), [0] * len(capacities))

        for result in results:
            idx = result['res_list']
            cap = result['capacity']
            cap_idx = capacity_index[cap]
            for k, v in result['averaged'].items():
                self.data[k][idx][cap_idx] = v

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"{timestamp}_zonesTcap_{dist}"
        self.RESULTS = f"{self.PATH}/results/{folder}"
        os.makedirs(f"{self.RESULTS}")
        print("Creating graphs...")
        self.create_graphs()
        print("Saving raw data...")
        self.save_results()
        self.cleanup()

    def create_files(self, dist):
        args = argparse.Namespace(
            files=1,
            min_nodes=self.NODES,
            max_nodes=self.NODES,
            min_x=-100,
            max_x=100,
            min_y=-100,
            max_y=100,
            min_weight=100.0,
            max_weight=250.0,
            min_speed=20.0,
            max_speed=60.0,
            verbose=False,
            distribution=dist
        )

        cm.create_datasets(args)

        src_base = f"{self.CWD}/Algorithm/Library/utils/datasets"
        dst_base = f"{self.PATH}/files"

        shutil.move(f"{src_base}/dataset1.txt", f"{dst_base}/file.txt")
        shutil.move(f"{src_base}/log.txt", f"{dst_base}/log.txt")

    def create_graphs(self):
        pltr = plotter.Plotter()
        x_values = list(range(1000, self.MAX_CAPACITY + 1, 100))
        ticks = list(range(1000, self.MAX_CAPACITY + 1, 1000))
        labels = ["Ascendent division", "Descendent division"]

        def save_plot(metric, title, ylabel, filename):
            plt_obj = pltr.plot_bench_results(self.data[metric], x_values, title, "Capacity", ylabel, labels, ticks)
            plt_obj.savefig(f"{self.RESULTS}/{filename}.png")
            plt_obj.close()

        save_plot("time_divide", "Time to divide a graph vs. number of nodes", "Time", "divide_time")
        save_plot("time_solve", "Time to solve a graph vs. capacity", "Time", "time_solve")
        save_plot("time_total", "Total time (divide + solve) vs. capacity", "Time", "time_total")
        save_plot("value", "Solution value vs. capacity", "Value", "value")
        save_plot("zones", "Number of zones vs. capacity", "Nº of zones", "zones")

        def save_bar(metric, title, filename):
            plt_obj = pltr.plot_distribution_bar_chart(data=self.data[metric],
                                                       title=title,
                                                       x_label="Nodes",
                                                       y_label="Occurrences",
                                                       labels=[f"{title} (DescDivi)", f"{title} (AscDivi)"])
            plt_obj.savefig(f"{self.RESULTS}/{filename}.png")
            plt_obj.close()

        save_bar("max_nodes", "Max nodes", "max_nodes")
        save_bar("avg_nodes", "Avg nodes", "avg_nodes")
        save_bar("min_nodes", "Min nodes", "min_nodes")

    def save_results(self):
        with open(f"{self.RESULTS}/results.json", "w") as f:
            json.dump(self.data, f)

    def zip_input_file(self):
        zip_path = os.path.join(self.RESULTS, "graphs.zip")
        files_dir = os.path.join(self.PATH, "files")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            file_path = os.path.join(files_dir, "file.txt")
            if os.path.exists(file_path):
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)

    def cleanup(self):
        os.remove(f"{self.PATH}/files/file.txt")
        os.remove(f"{self.PATH}/files/log.txt")


def main():
    bench = Benchmark()
    bench.run("u")
    bench = Benchmark()
    bench.run("n")
    bench = Benchmark()
    bench.run("k")


if __name__ == '__main__':
    main()
