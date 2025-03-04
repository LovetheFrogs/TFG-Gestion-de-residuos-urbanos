"""Auxiliary module to plot the results of running the genetic algorithm."""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    """Class containing different plotting methods."""
    def __init__(self):
        pass

    def plot_map(
        self,
        data: list[tuple[float, float]] | list[list[tuple[float, float]]],
        vrp: bool,
        center: tuple[float, float]
    ) -> plt:
        """Creates a plot of the nodes in a path.

        Shows how the best route joins each node to each other.

        Args:
            data: List of nodes. In vrp mode it will be a list of lists of 
                nodes (one for each zone).
            vrp: If the map to plot is for a TSP or a VRP instance.
            center: Coordinates of the central node.
        
        Returns:
            A plot of the nodes and how they are connected in a path.
        """
        if not vrp:
            plt.scatter(*zip(*data), marker='.', color='red')
            plt.plot(*zip(*data), linestyle='-', color='blue')
            plt.title('Best path found')
        else:
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.numOfVehicles)))
            for route in data:
                stops = [i for i in route]
                plt.plot(*zip(*stops), linestyle='-', color=next(color))
            plt.title('Best paths found')
        plt.plot(
            center[0], center[1], marker='x', markersize=10, color='green'
        )

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
