"""Auxiliary module to plot the results of running the genetic algorithm."""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class Plotter():
    """Class containing different plotting methods."""

    def __init__(self):
        pass

    def plot_points(self, data: list[tuple[float, float]],
                    center: tuple[float, float]) -> plt:
        """_summary_

        Args:
            data: A list of point coordinates
            center: The coordinates of the center node.

        Returns:
            A plot of the nodes that make up a graph.
        """
        plt.scatter(*zip(*data), marker='.', color='red')
        plt.plot(center[0], center[1], marker='x', markersize=10, color='green')
        plt.title('Bins')

        return plt

    def plot_map(self, data: list[tuple[float, float]],
                 center: tuple[float, float]) -> plt:
        """Creates a plot of the nodes in a path.

        Shows how the best route joins each node to each other.

        Args:
            data: List of nodes. In multi-zone mode it will be a list of lists 
                of nodes (one for each zone).
            center: The coordinates of the center node.
        
        Returns:
            A plot of the nodes and how they are connected in a path.
        """
        plt.scatter(*zip(*data), marker='.', color='red')
        plt.plot(*zip(*data), linestyle='-', color='blue')
        plt.title('Best path found')
        plt.plot(center[0], center[1], marker='x', markersize=10, color='green')

        return plt

    def plot_multiple_paths(self, data: list[list[tuple[float, float]]],
                            center: tuple[float, float]):
        """Plots multiple paths.

        Args:
            data: List containing different lists, each containing the 
                coordinates of a node in a path.
            center: The coordinates of the center node.

        Returns:
            A plot with all the paths that need to be represented.
        """
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        for path in data:
            col = next(color)
            plt.scatter(*zip(*path), marker='.', color=col)
            plt.plot(*zip(*path), linestyle='--', color=col)

        plt.title('Best paths found')
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

    def plot_zones(self, data: list[list[tuple[float, float]]],
                   center: tuple[float, float]) -> plt:
        """Plots all the nodes in a graph and identifies the zones they form.

        Args:
            data: A list of lists, each one containing the coordinates of the
                nodes in a zone.
            center: The coordinates of the center node.

        Returns:
            A plot containing all the nodes and their respective zones.
        """
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data))))
        x_limits = (0, 0)
        y_limits = (0, 0)
        for i, zone in enumerate(data):
            x_limits = (min(x_limits[0], min([x[0] for x in zone])),
                        max(x_limits[1], max([x[0] for x in zone])))
            y_limits = (min(y_limits[0], min([x[1] for x in zone])),
                        max(y_limits[1], max([x[1] for x in zone])))
            plt.scatter(*zip(*zone), marker='.', color=next(color))
            last_node = zone[0]
            first_node = data[i - 1][-1]
            middle_x = (last_node[0] + first_node[0]) / 2
            middle_y = (last_node[1] + first_node[1]) / 2
            dx = middle_x - center[0]
            dy = middle_y - center[1]
            scale = 1000
            end = (center[0] + dx * scale, center[1] + dy * scale)
            plt.plot((center[0], end[0]), (center[1], end[1]),
                     color='grey',
                     linestyle='--',
                     linewidth=1)

        plt.plot(center[0], center[1], marker='x', markersize=10, color='green')
        plt.title("Zone division")
        plt.xlim(x_limits[0] - 10 if x_limits[0] < 0 else x_limits[0] + 10,
                 x_limits[1] + 10 if x_limits[1] > 0 else x_limits[1] - 10)
        plt.ylim(y_limits[0] - 10 if y_limits[0] < 0 else y_limits[0] + 10,
                 y_limits[1] + 10 if y_limits[1] > 0 else y_limits[1] - 10)

        return plt
    
    def plot_bench_results(self, 
                           data: tuple[list, list], 
                           x_values: list, 
                           title: str,
                           x_label: str,
                           y_label: str, 
                           labels: list[str],
                           x_data_points: list[int]) -> plt:
        """Plots a 2-tuple of lists vs. x_values.

        Args:
            data: A 2-tuple of lists.
            x_values: List of all possible values for the x-axis.
            title: Title of the graph.
            y_label: Label of the y-axis.
            labels: List of labels of the plotted lines. 

        Returns:
            A plot containing all data.
        """
        plt.figure()
        plt.plot(x_values, data[0], label=labels[0])
        plt.plot(x_values, data[1], label=labels[1])
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(x_data_points, rotation=90)
        
        return plt
    
    def plot_distribution_bar_chart(self,
                                    data: tuple[list, list], 
                                    title: str,
                                    x_label: str,
                                    y_label: str,
                                    labels: list[str]) -> plt:
        """
        Plots a grouped bar chart showing the frequency of values in each list of the tuple.

        Args:
            data: Tuple of two lists (e.g., (desc_values, asc_values)).
            title: Title of the chart.
            x_label: Label for x-axis (e.g., "Nodes").
            y_label: Label for y-axis (e.g., "Occurrences").
            labels: Labels for the two datasets (e.g., ["DescDivi", "AscDivi"]).
        
        Returns:
            The matplotlib.pyplot object for further customization or saving.
        """
        counter1 = Counter(data[0])
        counter2 = Counter(data[1])

        all_keys = sorted(set(counter1.keys()).union(set(counter2.keys())))

        values1 = [counter1.get(k, 0) for k in all_keys]
        values2 = [counter2.get(k, 0) for k in all_keys]

        x = np.arange(len(all_keys))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, values1, width, label=labels[0], color='royalblue')
        rects2 = ax.bar(x + width/2, values2, width, label=labels[1], color='tomato')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90)
        ax.set_xticklabels(all_keys)
        ax.legend()

        return plt
