"""Auxiliary module to plot the results of running the genetic algorithm."""

import seaborn as sns
import matplotlib.pyplot as plt


class Plotter():
    """Class containing different plotting methods."""

    def __init__(self):
        pass

    def plot_map(self, data: list[tuple[float, float]] |
                 list[list[tuple[float, float]]], center: tuple[float,
                                                                float]) -> plt:
        """Creates a plot of the nodes in a path.

        Shows how the best route joins each node to each other.

        Args:
            data: List of nodes. In vrp mode it will be a list of lists of 
                nodes (one for each zone).
        
        Returns:
            A plot of the nodes and how they are connected in a path.
        """
        plt.scatter(*zip(*data), marker='.', color='red')
        plt.plot(*zip(*data), linestyle='-', color='blue')
        plt.title('Best path found')
        plt.plot(center[0], center[1], marker='x', markersize=10, color='green')

        return plt

    def plot_evolution(self, min: float, mean: float) -> plt:
        """Creates a plot of the evolution of the fitness values.
        
        Args:
            min: List of minimum fitness values.
            mean: List of average fitness values.

        Returns:
            A plot of the evolution of the fitness values.
        """
        sns.set_style("whitegrid")
        plt.plot(min, color='red')
        plt.plot(mean, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Min / Average Fitness')
        plt.title('Min and Average fitness evolution')

        return plt
