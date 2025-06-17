import os
import time
from ..problem import model
from ..problem import algorithms
from ..utils import utils


CWD = os.path.join(os.getcwd(), "Algorithm")


def run():
    g = model.Graph()
    g.populate_from_file(f"{CWD}/Library/tests/files/test3.txt", verbose=True)
    algo = algorithms.Algorithms(g)
    start = time.time()
    subgraphs, zones = algo.divide(1500, dir=f"{CWD}/Library/problem/plots", name="AscDivi")
    end = time.time()
    t1 = end - start
    start = time.time()
    subgraphs2, zones2 = algo.divide(1500, dir=f"{CWD}/Library/problem/plots", name="DesDivi", asc=True)
    end = time.time()
    t2 = end - start
    t1 *= 1000
    t2 *= 1000
    print(f"Zones -> AscDivi = {len(zones)} | DesDivi = {len(zones2)}")
    av1, av2, h1, h2, avgn1, avgn2, b1, b2 = 0, 0, 0, 0, 0, 0, 0, 0
    l1, l2, sm1, sm2 = float('inf'), float('inf'), float('inf'), float('inf')
    for s1, s2 in zip(subgraphs, subgraphs2):
        w1 = s1.total_weight()
        w2 = s2.total_weight()
        av1 += w1
        av2 += w2
        if w1 > h1:
            h1 = w1
        if w2 > h2:
            h2 = w2
        if w1 < l1:
            l1 = w1
        if w2 < l2:
            l2 = w2
        avgn1 += s1.nodes
        avgn2 += s2.nodes
        if s1.nodes > b1:
            b1 = s1.nodes
        if s2.nodes > b2:
            b2 = s2.nodes
        if s1.nodes < sm1:
            sm1 = s1.nodes
        if s2.nodes < sm2:
            sm2 = s2.nodes
        
    av1 /= len(zones)
    av2 /= len(zones2)
    avgn1 /= len(zones)
    avgn2 /= len(zones2)
    print(f"Avg weight -> AscDivi = {av1:.2f} | DesDivi = {av2:.2f}")
    print(f"Heaviest zone -> AscDivi = {h1:.2f} | DesDivi = {h2:.2f}")
    print(f"Lightest zone -> AscDivi = {l1:.2f} | DesDivi = {l2:.2f}")
    print(f"Avg zone size -> AscDivi = {int(avgn1)} | DesDivi = {int(avgn2)}")
    print(f"Biggest zone size -> AscDivi = {b1} | DesDivi = {b2}")
    print(f"Smallest zone size -> AscDivi = {sm1} | DesDivi = {sm2}")
    print("------------------------------------------------------")
    print(f"Time to compute -> AscDivi = {t1:.0f}ms | DesDivi = {t2:.0f}ms")
    
    

def run_avg_times():
    g = model.Graph()
    g.populate_from_file(f"{CWD}/Library/tests/files/test3.txt", verbose=False)
    t1, t2 = 0, 0
    t1l, t2l = [], []
    algo = algorithms.Algorithms(g)
    
    print("Running")
    utils.printProgressBar(0,
                    1000,
                    prefix="Running iterations:",
                    suffix=f"Complete (0/{1000})",
                    length=50,
                    show_eta=True)
    
    for i in range(1000):
        start = time.time()
        subgraphs, zones = algo.divide(1500, dir="False")
        end = time.time()
        t1 += (end - start) * 1000
        t1l.append((end - start) * 1000)
        start = time.time()
        subgraphs2, zones2 = algo.divide(1500, dir="False", asc=True)
        end = time.time()
        t2 += (end - start) * 1000
        t2l.append((end - start) * 1000)
        utils.printProgressBar(i + 1,
                                1000,
                                prefix="Running iterations:",
                                suffix=f"Complete ({i + 1}/{1000})",
                                length=50,
                                show_eta=True)


    t1 /= 1000
    t2 /= 1000
    print()
    print(f"Avg time after 1000 executions -> AscDivi = {t1:.2f}ms | DesDivi = {t2:.2f}ms")
    

def main():
    run()
    run_avg_times()


if __name__ == '__main__':
    main()
