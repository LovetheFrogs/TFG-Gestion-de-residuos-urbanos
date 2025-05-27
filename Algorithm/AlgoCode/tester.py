from problem import model, algorithms
from utils import plotter
import os

if __name__ == "__main__":
    cwd = os.getcwd()
    g = model.Graph()
    g.populate_from_file(f'{cwd}/tests/files/test4.txt')
    algo = algorithms.Algorithms(g)
    _, _ = algo.divide(100000, dir=f"{cwd}/problem/plots", name="50")
    p, _ = algo.nearest_neighbor(dir=f"{cwd}/plots2", name="NN50")
    _, v2optrand = algo.run_two_opt(dir=f"{cwd}/plots2", name="2opt50rand")
    _, v2opt = algo.run_two_opt(path=p, dir=f"{cwd}/plots2", name="2opt50")
    _, vsarand = algo.run_sa(dir=f"{cwd}/plots2", name="sa50rand")
    _, vsa = algo.run_sa(path=p, dir=f"{cwd}/plots2", name="sa50")
    
    
    print(f"2optrand = {v2optrand} | 2opt = {v2opt}")
    print(f"SArand = {vsarand} | SA = {vsa}")
    