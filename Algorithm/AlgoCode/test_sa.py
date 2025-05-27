import os
import json
import itertools
import heapq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from problem import model, algorithms
import numpy as np
import multiprocessing
from collections import OrderedDict

# ----- Setup -----
g = model.Graph()
g.populate_from_file(f'{os.getcwd()}/tests/files/test4.txt')
algo = algorithms.Algorithms(g)
p, _ = algo.nearest_neighbor()
p_copy = p.copy()

niter_options = list(range(10000, 1000001, 10000))               # 100
mstag_options = list(range(1500, 100001, 5000))                  # 20
temperature_options = list(range(1000, 10001, 1000))             # 10
alpha_options = [round(a, 4) for a in np.arange(0.85, 1.0, 0.001)]  # 1500

CACHE_FILE = "results_cache.json"
MAX_CACHE_ENTRIES = 100_000
best = []

# ----- Load cache (last 100000 only) -----
cache = OrderedDict()
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        raw_cache = json.load(f)
        # Truncate to most recent MAX_CACHE_ENTRIES
        for k in list(raw_cache)[-MAX_CACHE_ENTRIES:]:
            cache[k] = raw_cache[k]

# ----- Generate combos -----
all_combos = list(itertools.product(niter_options, mstag_options, temperature_options, alpha_options))
filtered_combos = [combo for combo in all_combos if str(combo) not in cache]
total_combos = len(filtered_combos)
print(f"Total new combinations to test: {total_combos:,}")

# ----- Worker -----
def run_combo(combo, path_copy):
    niter, mstag, temp, alpha = combo
    g_local = model.Graph()
    g_local.populate_from_file(f'{os.getcwd()}/tests/files/test4.txt')
    algo_local = algorithms.Algorithms(g_local)
    _, value = algo_local._simulated_annealing(path_copy.copy(), niter, mstag, False, temp, alpha)
    return combo, value

# ----- Parallel Execution with Progress -----
with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    futures = {
        executor.submit(run_combo, combo, p_copy): combo
        for combo in filtered_combos
    }

    with tqdm(total=total_combos, desc="Testing", unit="combo", dynamic_ncols=True) as pbar:
        for future in as_completed(futures):
            combo = futures[future]
            try:
                (niter, mstag, temp, alpha), value = future.result()
                key = str(combo)

                # Maintain last MAX_CACHE_ENTRIES in cache
                if key not in cache:
                    if len(cache) >= MAX_CACHE_ENTRIES:
                        cache.popitem(last=False)  # Remove oldest
                    cache[key] = value

                if len(best) < 10:
                    heapq.heappush(best, (-value, (niter, mstag, temp, alpha, value)))
                else:
                    heapq.heappushpop(best, (-value, (niter, mstag, temp, alpha, value)))

                pbar.update(1)

            except Exception as e:
                print(f"Error in combo {combo}: {e}")

            # Save periodically
            if pbar.n % 1000 == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f)

# ----- Final Save -----
with open(CACHE_FILE, "w") as f:
    json.dump(cache, f)

# ----- Top 10 -----
print("\nTop 10 parameter combinations:")
best.sort(reverse=True)
for _, (niter, mstag, temp, alpha, val) in best:
    print(f"niter: {niter}, mstag: {mstag}, temp: {temp}, alpha: {alpha:.4f} -> TSP value: {val:.4f}")
