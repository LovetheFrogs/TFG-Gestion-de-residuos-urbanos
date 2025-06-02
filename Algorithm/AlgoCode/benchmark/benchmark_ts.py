import os
import random
import numpy as np
import heapq
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import itertools
from problem import model, algorithms


def run_combo(full_combo):
    try:
        combo, path, file_path = full_combo
        mstag, niter, sigma = combo

        from problem import model, algorithms
        g_local = model.Graph()
        g_local.populate_from_file(file_path)
        algo_local = algorithms.Algorithms(g_local)
        _, value = algo_local._tabu_search(path.copy(), niter, mstag, False, sigma)

        return value, (mstag, niter, sigma)
    except Exception as e:
        return -1e9, ("ERROR", str(e))


def generate_sampled_combos(sample_size=4000):
    mstag_options = list(range(100, 1001, 100))
    niter_options = list(range(100000, 500001, 100000))
    sigma_options = list(range(100000, 300001, 100000))

    all_combos = list(itertools.product(mstag_options, niter_options, sigma_options))
    print(f"Total possible combinations: {len(all_combos):,}")

    sampled_combos = random.sample(all_combos, sample_size)
    return sampled_combos


def save_results(file_name, best_combinations):
    output_path = os.path.join("results", f"results_{file_name}")
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Top 10 results for file: {file_name}\n\n")
        for mstag, niter, sigma, val in best_combinations:
            f.write(f"mstag: {mstag}, temp: {niter}, alpha: {sigma:.4f} -> TSP value: {val:.4f}\n")


def benchmark(f, sampled_combos):
    print(f"\nTesting for file {f}")
    file_path = os.path.join(os.getcwd(), 'tests', 'files', f)
    g = model.Graph()
    g.populate_from_file(file_path)
    algo = algorithms.Algorithms(g)
    p, _ = algo.nearest_neighbor(dir=os.getcwd(), name="aux")

    full_combo = ((combo, p, file_path) for combo in sampled_combos)

    max_cores = 4
    print(f"Running on {max_cores} cores...")

    best = []

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for value, params in tqdm(
            executor.map(run_combo, full_combo, chunksize=5),
            total=len(sampled_combos),
            desc="Testing",
            unit="combo",
            dynamic_ncols=True
        ):
            if isinstance(params[0], str) and params[0] == "ERROR":
                print(f"Error in worker: {params[1]}")
                continue

            mstag, niter, sigma = params
            result = (mstag, niter, sigma, value)

            if len(best) < 10:
                heapq.heappush(best, (-value, result))
            else:
                heapq.heappushpop(best, (-value, result))

    print("\nTop 10 combinations:")
    best.sort(reverse=True)
    for _, (mstag, niter, sigma, val) in best:
        print(f"mstag: {mstag}, niter: {niter}, sigma: {sigma:.4f} -> TSP value: {val:.4f}")

    save_results(f, [item[1] for item in best])


def main():
    SAMPLE_SIZE = 50
    random.seed(42)
    sampled_combos = generate_sampled_combos(sample_size=SAMPLE_SIZE)

    test_files = ["test150.txt", "test200.txt"]
    for f in test_files:
        benchmark(f, sampled_combos)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
