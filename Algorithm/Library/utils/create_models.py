import argparse
import random
import math
from typing import Tuple, Set
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
import random
from utils import utils

NEW_LINE = "\n"
DECORATOR = "+------------------------------------------------------------+"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets.")
    parser.add_argument("-f", "--files", type=int, default=1)
    parser.add_argument("-n", "--min_nodes", type=int, default=200)
    parser.add_argument("-N", "--max_nodes", type=int, default=200)
    parser.add_argument("-x", "--min_x", type=int, default=-100)
    parser.add_argument("-X", "--max_x", type=int, default=100)
    parser.add_argument("-y", "--min_y", type=int, default=-100)
    parser.add_argument("-Y", "--max_y", type=int, default=100)
    parser.add_argument("-w", "--min_weight", type=float, default=100.0)
    parser.add_argument("-W", "--max_weight", type=float, default=250.0)
    parser.add_argument("-s", "--min_speed", type=float, default=20.0)
    parser.add_argument("-S", "--max_speed", type=float, default=60.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--distribution", choices=["u", "n", "k"], default="u",
                        help="Coordinate distribution: u=uniform, n=normal, k=clustered")
    return parser.parse_args()

def generate_nodes(args) -> Tuple[Set[int], int, float, str]:
    num_nodes = random.randint(args.min_nodes, args.max_nodes)
    nodes = set(range(num_nodes))
    node_lines = []
    weights = []

    def sample_coord(min_val, max_val):
        if args.distribution == "u":
            return random.randint(min_val, max_val)
        elif args.distribution == "n":
            mu = (min_val + max_val) / 2
            sigma = (max_val - min_val) / 6
            return max(min(int(random.gauss(mu, sigma)), max_val), min_val)
        else:
            raise ValueError("sample_coord should not be called for 'k' distribution.")

    if args.distribution == "k":
        cluster_count = max(1, num_nodes // 10)
        cluster_centers = [
            (random.randint(args.min_x, args.max_x), random.randint(args.min_y, args.max_y))
            for _ in range(cluster_count)
        ]

        for node in nodes:
            cx, cy = random.choice(cluster_centers)
            spread = (args.max_x - args.min_x) // 20
            x = max(min(int(random.gauss(cx, spread)), args.max_x), args.min_x)
            y = max(min(int(random.gauss(cy, spread)), args.max_y), args.min_y)
            weight = round(random.uniform(args.min_weight, args.max_weight), 2)
            node_lines.append(f"{node} {weight:.2f} {x} {y}")
            weights.append(weight)

    else:
        for node in nodes:
            weight = round(random.uniform(args.min_weight, args.max_weight), 2)
            x = sample_coord(args.min_x, args.max_x)
            y = sample_coord(args.min_y, args.max_y)
            node_lines.append(f"{node} {weight:.2f} {x} {y}")
            weights.append(weight)

    node_data = f"{num_nodes}{NEW_LINE}" + NEW_LINE.join(node_lines) + NEW_LINE
    return nodes, num_nodes, sum(weights), node_data

def generate_edges(nodes: Set[int], args) -> Tuple[int, str]:
    nodes_list = list(nodes)
    edge_data = []
    edges = 0
    edges_added = set()
    tot_edges = len(nodes) * (len(nodes) - 1)

    if args.verbose:
        utils.printProgressBar(0, tot_edges, prefix="Progress:", suffix="Complete", length=50)

    for node1 in nodes_list:
        for node2 in nodes_list:
            if node1 == node2 or (node1, node2) in edges_added:
                continue
            speed = round(random.uniform(args.min_speed, args.max_speed), 1)
            edge_data.append(f"{speed} {node1} {node2}")
            edge_data.append(f"{speed} {node2} {node1}")
            edges += 2
            edges_added.update([(node1, node2), (node2, node1)])

            if args.verbose:
                utils.printProgressBar(edges, tot_edges,
                                       prefix="Progress:",
                                       suffix=f"Complete ({edges}/{tot_edges})",
                                       length=50)
    return edges, f"{edges}{NEW_LINE}" + NEW_LINE.join(edge_data) + NEW_LINE

def create_log(log_data, total_nodes, total_edges, total_density, total_weight, output_path, file_count):
    log = f"Generated {file_count} datasets.{NEW_LINE}{DECORATOR}{NEW_LINE}"
    for k, (nodes, edges, density, weight) in enumerate(log_data, start=1):
        log += (f"dataset{k}{NEW_LINE}    |- Nodes: {nodes}{NEW_LINE}"
                f"    |- Edges: {edges}{NEW_LINE}    |- Density: {density:.4f}"
                f"{NEW_LINE}    |- Weight: {weight:.2f}{NEW_LINE}{DECORATOR}{NEW_LINE}")
    log += (f"Averages{NEW_LINE}    |- Nodes: {total_nodes / file_count:.2f}"
            f"{NEW_LINE}    |- Edges: {total_edges / file_count:.2f}"
            f"{NEW_LINE}    |- Density: {total_density / file_count:.4f}"
            f"{NEW_LINE}    |- Weight: {total_weight / file_count:.2f}{NEW_LINE}")

    with open(os.path.join(output_path, "log.txt"), "w") as f:
        f.write(log)

def create_datasets(args):
    output_path = os.path.join(os.getcwd(), "Algorithm/Library/utils/datasets")
    os.makedirs(output_path, exist_ok=True)

    total_nodes = total_edges = total_density = total_weight = 0
    log = []

    if args.verbose:
        print(f"Generating {args.files} files with '{args.distribution}' distribution...")

    for k in range(1, args.files + 1):
        if args.verbose:
            print(f"Generating file {k}")
        nodes, node_count, weight, node_data = generate_nodes(args)
        edge_count, edge_data = generate_edges(nodes, args)

        density = edge_count / (node_count * (node_count - 1))
        log.append((node_count, edge_count, density, weight))
        total_nodes += node_count
        total_edges += edge_count
        total_weight += weight
        total_density += density

        with open(os.path.join(output_path, f"dataset{k}.txt"), "w") as f:
            f.write(node_data + edge_data)

    create_log(log, total_nodes, total_edges, total_density, total_weight, output_path, args.files)

    if args.verbose:
        print("Generation complete.")

def main():
    args = parse_args()
    create_datasets(args)

if __name__ == "__main__":
    main()
