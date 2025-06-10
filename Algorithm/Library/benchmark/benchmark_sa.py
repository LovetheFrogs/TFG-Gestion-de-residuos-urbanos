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
        mstag, temp, alpha = combo

        from problem import model, algorithms
        g_local = model.Graph()
        g_local.populate_from_file(file_path)
        algo_local = algorithms.Algorithms(g_local)
        _, value = algo_local._simulated_annealing(path.copy(), mstag, False, temp, alpha)

        return value, (mstag, temp, alpha)
    except Exception as e:
        return -1e9, ("ERROR", str(e))


def generate_sampled_combos(sample_size=4000):
    mstag_options = list(range(1000, 10001, 1000))
    temperature_options = list(range(1000, 10001, 1000))
    alpha_options = [round(a, 4) for a in np.arange(0.9, 0.999, 0.001)]

    all_combos = list(itertools.product(mstag_options, temperature_options, alpha_options))
    print(f"Total possible combinations: {len(all_combos):,}")

    sampled_combos = random.sample(all_combos, sample_size)
    return sampled_combos


def save_results(file_name, best_combinations):
    output_path = os.path.join("results", f"results_{file_name}")
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Top 10 results for file: {file_name}\n\n")
        for mstag, temp, alpha, val in best_combinations:
            f.write(f"mstag: {mstag}, temp: {temp}, alpha: {alpha:.4f} -> TSP value: {val:.4f}\n")


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
            executor.map(run_combo, full_combo, chunksize=20),
            total=len(sampled_combos),
            desc="Testing",
            unit="combo",
            dynamic_ncols=True
        ):
            if isinstance(params[0], str) and params[0] == "ERROR":
                print(f"Error in worker: {params[1]}")
                continue

            mstag, temp, alpha = params
            result = (mstag, temp, alpha, value)

            if len(best) < 10:
                heapq.heappush(best, (-value, result))
            else:
                heapq.heappushpop(best, (-value, result))

    print("\nTop 10 combinations:")
    best.sort(reverse=True)
    for _, (mstag, temp, alpha, val) in best:
        print(f"mstag: {mstag}, temp: {temp}, alpha: {alpha:.4f} -> TSP value: {val:.4f}")

    save_results(f, [item[1] for item in best])


def main():
    SAMPLE_SIZE = 4000
    random.seed(42)
    sampled_combos = generate_sampled_combos(sample_size=SAMPLE_SIZE)

    test_files = ["test25.txt", "test4.txt", "test50.txt", "test75.txt", "test100.txt", "test150.txt", "test200.txt"]
    for f in test_files:
        benchmark(f, sampled_combos)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
