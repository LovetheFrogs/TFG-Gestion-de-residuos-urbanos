import datetime
import glob
import multiprocessing
import os
import sys
import json
import argparse
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import zipfile
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from ..utils import create_models as cm
from ..utils import plotter
from ..utils import utils
from ..problem.model import Graph

def worker_run_divide_solve(file_index, path, cap, asc, repetitions):
    import time
    import statistics
    from ..problem.algorithms import Algorithms
    from ..problem.model import Graph

    g = Graph()
    g.populate_from_file(f"{path}/files/file{file_index}.txt")

    temp_metrics = {
        "time_divide": [],
        "time_solve": [],
        "time_total": [],
        "value": [],
        "zones": [],
        "max_nodes": [],
        "avg_nodes": [],
        "min_nodes": []
    }

    for _ in range(repetitions):
        algo = Algorithms(g)
        start = time.time()
        subgraphs, z = algo.divide(zone_weight=cap, dir="False", asc=asc)
        end = time.time()
        time_divide = (end - start) * 1000
        temp_metrics["time_divide"].append(time_divide)

        tot_runtime = 0
        tot_value = 0
        for sg in subgraphs:
            algo = Algorithms(sg)
            start2 = time.time()
            _, v = algo.run(dir="False")
            end2 = time.time()
            tot_runtime += (end2 - start2)
            tot_value += v

        temp_metrics["time_solve"].append(tot_runtime * 1000)
        temp_metrics["time_total"].append(tot_runtime * 1000 + time_divide)
        temp_metrics["value"].append(tot_value)
        temp_metrics["zones"].append(len(z))

        node_counts = [len(sg) for sg in subgraphs]
        temp_metrics["min_nodes"].append(min(node_counts))
        temp_metrics["max_nodes"].append(max(node_counts))
        temp_metrics["avg_nodes"].append(sum(node_counts) / len(node_counts))

    return (file_index, asc, {
        k: int(statistics.mean(v)) if k.startswith("avg_nodes") else statistics.mean(v)
        for k, v in temp_metrics.items()
    })


class Benchmark():
    def __init__(self):
        self.NUM_FILES = 250
        self.CAPACITY = 1500
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
            "zones": ([], [])
        }
        self.g = Graph()

    def run(self, dist):
        print("Creating files...")
        self.create_files(dist)

        print("Running benchmark...")
        with tqdm(total=(self.NUM_FILES + 1) * 2, desc="Processing", ncols=90) as pbar:
            with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()//4)) as executor:
                futures = []
                for i in range(self.NUM_FILES + 1):
                    for asc in [False, True]:
                        futures.append(executor.submit(
                            worker_run_divide_solve,
                            i, self.PATH, self.CAPACITY, asc, 10
                        ))

                for future in as_completed(futures):
                    try:
                        file_index, asc, result = future.result()
                        res_list = int(asc)
                        for k, v in result.items():
                            self.data[k][res_list].append(v)
                    except Exception as e:
                        print("Error in worker:", e)
                    pbar.update(1)

        for k in self.data:
            assert len(self.data[k][0]) == self.NUM_FILES + 1, f"{k}[0] is incorrect length"
            assert len(self.data[k][1]) == self.NUM_FILES + 1, f"{k}[1] is incorrect length"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"{timestamp}_zonesGsize_{dist}"
        self.RESULTS = f"{self.PATH}/results/{folder}"
        os.makedirs(f"{self.RESULTS}")

        print("Creating graphs...")
        self.create_graphs()
        print("Saving raw data...")
        self.save_results()
        print("Saving graph files...")
        self.zip_input_files()
        self.cleanup()

    def create_files(self, dist):
        for i in range(self.NUM_FILES + 1):
            nodes = 25 + i
            args = argparse.Namespace(
                files=1,
                min_nodes=nodes,
                max_nodes=nodes,
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

            shutil.move(f"{src_base}/dataset1.txt", f"{dst_base}/file{i}.txt")

        shutil.move(f"{src_base}/log.txt", f"{dst_base}/log.txt")

    def create_graphs(self):
        pltr = plotter.Plotter()
        x_range = list(range(25, 276, 1))
        ticks = range(25, 276, 25)
        labels = ["Ascendent division", "Descendent division"]

        def save_plot(metric, title, ylabel, filename):
            plt_obj = pltr.plot_bench_results(self.data[metric], x_range, title, "Number of nodes", ylabel, labels, ticks)
            plt_obj.savefig(f"{self.RESULTS}/{filename}.png")
            plt_obj.close()

        save_plot("time_divide", "Time to divide a graph vs. number of nodes", "Time", "divide_time")
        save_plot("time_solve", "Time to solve a graph vs. number of nodes", "Time", "time_solve")
        save_plot("time_total", "Total time (divide + solve) vs. number of nodes", "Time", "time_total")
        save_plot("value", "Solution value vs. number of nodes", "Value", "value")
        save_plot("zones", "Number of zones vs. number of nodes", "Nº of zones", "zones")

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
        
    def zip_input_files(self):
        zip_path = os.path.join(self.RESULTS, "graphs.zip")
        files_dir = os.path.join(self.PATH, "files")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i in range(self.NUM_FILES):
                file_path = os.path.join(files_dir, f"file{i}.txt")
                if os.path.exists(file_path):
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)

    def save_results(self):
        with open(f"{self.RESULTS}/results.json", "w") as f:
            json.dump(self.data, f)

    def cleanup(self):
        for i in range(self.NUM_FILES + 1):
            os.remove(f"{self.PATH}/files/file{i}.txt")
        os.remove(f"{self.PATH}/files/log.txt")


def main():
    bench = Benchmark()
    bench.run("u")
    bench = Benchmark()
    bench.run("n")
    bench = Benchmark()
    bench.run("k")


if __name__ == '__main__':
    sstart = time.time()
    main()
    send = time.time()
    timestamp = send - sstart
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print("Elapsed time:", formatted_time)
