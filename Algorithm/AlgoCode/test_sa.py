import os
import random
import numpy as np
import heapq
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from problem import model, algorithms
import itertools

def run_combo(combo_with_path_and_file):
    try:
        combo, path, file_path = combo_with_path_and_file
        niter, mstag, temp, alpha = combo

        # Local import for subprocess safety
        from problem import model, algorithms
        g_local = model.Graph()
        g_local.populate_from_file(file_path)
        algo_local = algorithms.Algorithms(g_local)
        _, value = algo_local._simulated_annealing(path.copy(), niter, mstag, False, temp, alpha)

        return value, (niter, mstag, temp, alpha)
    except Exception as e:
        return -1e9, ("ERROR", str(e))

def benchmark(f):
    print(f"Testing for file {f}")
    file_path = os.path.join(os.getcwd(), 'tests', 'files', f)
    g = model.Graph()
    g.populate_from_file(file_path)
    algo = algorithms.Algorithms(g)
    p, _ = algo.nearest_neighbor(dir=os.getcwd(), name="aux")

    # Parameter space
    niter_options = list(range(100000, 1000001, 100000))              
    mstag_options = list(range(10000, 100001, 10000))                    
    temperature_options = list(range(1000, 10001, 1000))               
    alpha_options = [round(a, 4) for a in np.arange(0.9, 0.9999, 0.0001)]

    all_combos = list(itertools.product(niter_options, mstag_options, temperature_options, alpha_options))
    print(f"Total possible combinations: {len(all_combos):,}")

    # Randomly sample combinations
    SAMPLE_SIZE = 4000
    sampled_combos = random.sample(all_combos, SAMPLE_SIZE)
    combo_with_paths = ((combo, p, file_path) for combo in sampled_combos)

    max_cores = 7  # Adjust to reduce thermal throttling
    print(f"Running on {max_cores} cores...")

    best = []

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for value, params in tqdm(
            executor.map(run_combo, combo_with_paths, chunksize=20),
            total=SAMPLE_SIZE,
            desc="Testing",
            unit="combo",
            dynamic_ncols=True
        ):
            if isinstance(params[0], str) and params[0] == "ERROR":
                print(f"Error in worker: {params[1]}")
                continue

            niter, mstag, temp, alpha = params
            if len(best) < 10:
                heapq.heappush(best, (-value, (niter, mstag, temp, alpha, value)))
            else:
                heapq.heappushpop(best, (-value, (niter, mstag, temp, alpha, value)))

    print("\nTop 10 combinations:")
    best.sort(reverse=True)
    for _, (niter, mstag, temp, alpha, val) in best:
        print(f"niter: {niter}, mstag: {mstag}, temp: {temp}, alpha: {alpha:.4f} -> TSP value: {val:.4f}")


def main():
    test_files = ["test25.txt", "test4.txt", "test75.txt", "test100.txt", "test150.txt", "test200.txt"]
    for f in test_files:
        benchmark(f)
        

if __name__ == "__main__":
    main()
