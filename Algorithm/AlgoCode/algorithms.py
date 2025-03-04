"""Algorithms used to calculate a path in a graph."""

import random
import statistics
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
import plotter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import Graph


class Algorithms():
    """Class containing the diferent path-finding algorithms.
    
    Args:
        graph: The graph object where the paths will be calculated on.
    """

    def __init__(self, graph: 'Graph'):
        self.graph = graph
        self.convert = None
        self.truck_capacity = None
        self.max_time = 8

    def evaluate(self, individual: list[int]) -> float:
        # total time = edge time + 2minutes to pick up bin.
        total_value = 0
        total_time = 0
        current = self.graph.center
        for idx in individual:
            aux = self.graph.get_node(idx)
            curr_edge = self.graph.get_edge(
                current, aux)
            total_value += curr_edge.value
            total_time += curr_edge.time + 0.03
            if total_time > self.max_time: return 100000
            current = aux
        total_value += self.graph.get_edge(current, self.graph.center).value
        penalty = sum(
            self.graph.get_node(node).weight *
            (len(individual) - i) for i, node in enumerate(individual))
        return (total_value + 0.2 * penalty + 0.5 * total_time)

    def evaluate_tsp(self, individual: list[int]) -> tuple[float, ...]:
        """Evaluates the objective function value for a path.
        
        The algorithms used for this problem are genetic algorithms and, as 
        such, try to minimize/maximize the value of a function to find a 
        solution to a problem. In this case, the problem is finding the path
        in a graph that optimizes a series of objectives. Those individual 
        objectives form an objective function. The ``evaluate`` function checks
        the result of evaluating said objective function for a given path.
        
        Currently, the objective function gives the cost of the path, which is
        the sum of the values of the edges that form the path, and a penalty,
        whose objective is to try and visit higher-weighted nodes later in the
        path, as running for a longer distance with a heavier load increases
        the maintenance cost of the truck.
        
        Args:
            individual: The path to evaluate.

        Returns:
            A tuple containing the value of the objective function.
        """
        return (self.evaluate([self.convert[node] for node in individual])),

    def evaluate_vrp(self, individual: list[int]) -> tuple[float, ...]:
        """Evaluates the objective function value for a path.

        This version of the ``evaluate`` function is the one used for the 
        Vehicle Routing Problem (VRP), as while the individuals are represented
        similarly, the fitness function for it should take into account both 
        the total lenght of the path, as well as the lenght of the longest path

        The algorithms used for this problem are genetic algorithms and, as 
        such, try to minimize/maximize the value of a function to find a 
        solution to a problem. In this case, the problem is finding the path
        in a graph that optimizes a series of objectives. Those individual 
        objectives form an objective function. The ``evaluate`` function checks
        the result of evaluating said objective function for a given path.

        Currently, the objective function gives the cost of the paths, which is
        the sum of the values of the edges that form the path, and a penalty,
        whose objective is to try and visit higher-weighted nodes later in the
        path, as running for a longer distance with a heavier load increases
        the maintenance cost of the truck. It also gives paths with similarly
        weighted zones a better finess value, and tries to minimize the 
        maximum value of a zone, as well as giving the maximum penalty to zones
        whose total weight exceeds the maximum weight a truck can pick up (this
        one is a hard constraint we have to consider).

        Args:
            individual: The path to evaluate.

        Returns:
            A tuple containing the value of the objective function.
        """
        zones = self.graph.extract_zones(individual)
        max_value = -1
        total_value = 0
        weights = []
        for zone in zones:
            new_value = 0
            total_weight = sum(
                self.graph.get_node(idx).weight for idx in zone)
            if total_weight > self.truck_capacity or not zone:
                new_value = 100000
            elif zone: 
                new_value = self.evaluate(zone)
            weights.append(total_weight)
            if new_value > max_value:
                max_value = new_value
            total_value += new_value

        total_value += 0.5 * max_value * len(zones)
        total_value += 1.0 * self.zone_likeness(weights)

        return (total_value,)

    def zone_likeness(self, zone_weights: list[float]) -> float:
        """Returns the likeliness index of a list of zones.
        
        The zone likeliness is defined as how similar are zones between them.
        Currently, all zones that differ 20% or more from the average zone's
        weight are considered not similar. To calculate the likeliness factor
        of a list of zones, we first calculate the average weight, then, the 
        diference percentage between each zone's weight and the average, and
        then we apply the likeliness function to each percentage (in base 1)
        to get the likeliness factor.

        Our likeliness function is defined as: $likeliness(x) = 100 · (5x)^2$,
        as using a cuadratic function to get the likeliness values allows us 
        to get a big jump in results when x > 0.2.
        
        For example, for a list of zones with weights = [954, 870, 642, 326, 
        1.250, 2.000, 790, 825], the average weight would be 957, the diference
        percentages are obtained from $|1 - 957/x|$, and are [0.004, 0.091, 
        0.329, 0.66, 0.306, 1.09, 0.175, 0.138] and the likeliness factors
        are [0.04, 20.7, 270.6, 1089, 239.1, 2500, 76.56, 47.61], thus giving 
        us a total likeliness factor of 4.243,61, making this zone division 
        highly unlikely to pass on to the next generation.
        """
        avg_weight = statistics.fmean(zone_weights) + 1
        avg_difs = [abs(1 - (z / avg_weight)) for z in zone_weights]
        likeliness_factors = [100 * pow(5 * ad, 2) for ad in avg_difs]

        return sum(likeliness_factors)

    def _define_creator(self) -> creator:
        """Defines a deap creator for the genetic algorithms.
        
        The ``deap.creator`` module is part of the DEAP framework and it's used
        to extend existing classes, adding new functionalities to them. This
        function extracts the ``creator`` instantiation from the ``run_ga_tsp``
        function so the code is easier to read and follow.
        
        Inside the ``creator`` object is where the objective of the genetic
        algorithm is defined, as well as what will the individuals be like.
        In this case, the objective is to minimize the value of the objective
        function, and the individuals are lists of integers, containing the 
        indices of the nodes of the graph in the order they will be visited.
        
        Returns:
            The creator defined for the genetic algorithm.
        """
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual",
                       list,
                       typecode='i',
                       fitness=creator.FitnessMin)

        return creator

    def _define_toolbox(self) -> base.Toolbox:
        """Defines a deap toolbox for the genetic algorithms.
        
        The ``deap.base.createor`` module is part of the DEAP framework. It's 
        used as a container for functions, and enables the creation of new
        operators by customizing existing ones. This function extracts the
        ``toolbox`` instantiation from the ``run_ga_tsp`` function so the code
        is easier to read and follow. 
        
        In the ``toolbox`` object is where the functions used by the genetic
        algorithm are defined, such as the evaluation, selection, crossover
        and mutation functions.

        Returns:
            The toolbox defined for the genetic algorithm.
        """
        nodes = [node.index for node in self.graph.node_list]
        genes = [i for i in range(len(nodes))]
        self.convert = {i: node for i, node in enumerate(nodes)}

        toolbox = base.Toolbox()
        toolbox.register("random_order", random.sample, genes, len(nodes))
        toolbox.register("individual_creator", tools.initIterate,
                         creator.Individual, toolbox.random_order)
        toolbox.register("population_creator", tools.initRepeat, list,
                         toolbox.individual_creator)

        toolbox.register("evaluate", self.evaluate_tsp)
        toolbox.register("select", tools.selTournament, tournsize=2)

        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate",
                         tools.mutShuffleIndexes,
                         indpb=1.0 / self.graph.nodes)

        return toolbox

    def _define_toolbox_vrp(self, agent_num: int) -> base.Toolbox:
        """Defines a deap toolbox for the genetic algorithms.
        
        The ``deap.base.createor`` module is part of the DEAP framework. It's 
        used as a container for functions, and enables the creation of new
        operators by customizing existing ones. This function extracts the
        ``toolbox`` instantiation from the ``run_ga_tsp`` function so the code
        is easier to read and follow. 
        
        In the ``toolbox`` object is where the functions used by the genetic
        algorithm are defined, such as the evaluation, selection, crossover
        and mutation functions.

        Args:
            agent_num: Number of agents (trucks).

        Returns:
            The toolbox defined for the genetic algorithm.
        """
        toolbox = base.Toolbox()
        toolbox.register("random_order", random.sample,
                         range(self.graph.nodes + agent_num - 1),
                         self.graph.nodes + agent_num - 1)
        toolbox.register("individual_creator", tools.initIterate,
                         creator.Individual, toolbox.random_order)
        toolbox.register("population_creator", tools.initRepeat, list,
                         toolbox.individual_creator)

        toolbox.register("evaluate", self.evaluate_vrp)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate",
                         tools.cxUniformPartialyMatched,
                         indpb=2.0 / (self.graph.nodes + agent_num))
        toolbox.register("mutate",
                         tools.mutShuffleIndexes,
                         indpb=1.0 / ((self.graph.nodes + agent_num)))

        return toolbox

    def _define_ga(self, toolbox: base.Toolbox,
                   pop_size: int) -> tuple[list, dict, list]:
        """Defines the attributes for the Generic Algorithm.
        
        The function defines the population, statistics and hall of fame for
        the Genetic Algorithm designed to solve the Traveling Salesman Problem.

        Args:
            toolbox: The toolbox for the genetic algorithm.
            pop_size: The size of the population.

        Returns:
            A tuple containing the population, statistics and hall of fame.
        """
        population = toolbox.population_creator(n=pop_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        hof = tools.HallOfFame(30)

        return population, stats, hof

    def eaSimpleWithElitism(self,
                            population,
                            toolbox,
                            cxpb,
                            mutpb,
                            ngen,
                            stats=None,
                            halloffame=None,
                            verbose=__debug__):
        """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
        halloffame is used to implement an elitism mechanism. The individuals contained in the
        halloffame are directly injected into the next generation and are not subject to the
        genetic operators of selection, crossover and mutation.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is None:
            raise ValueError("halloffame parameter must not be empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - hof_size)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population:
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def run_ga_tsp(self,
                   ngen: int = 100,
                   cxpb: float = 0.9,
                   mutpb: float = 0.1,
                   pop_size: int = 200,
                   dir: str | None = None,
                   idx: int = 0,
                   vrb: bool = True) -> tuple[list[int], float]:
        """Runs the Genetic Algorithm for the Traveling Salesman Problem.
        
        This function calls the wrapper functions that define the creator, 
        toolbox and the attributes for the Genetic Algorithm designed to solve
        the Traveling Salesman Problem. It then runs the Genetic Algorithm and 
        returns the best path found and its total value, while also calling the
        wrapper function to plot the results.

        Args:
            ngen (optional): The number of generations. Defaults to 100.
            cxpb (optional): The mating probability. Defaults to 0.9.
            mutpb (optional): The mutation probability. Defaults to 0.1.
            pop_size (optional): The size of the population. Defaults to 200.
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            idx (optional): The index for the plot to save. Defaults to 0.
            vrb: (optional): Run the algorithm in verbose or non-verbose mode.
                Defaults to True.

        Returns:
            A tuple containing the best path found and its total value.
        """
        creator = self._define_creator()
        toolbox = self._define_toolbox()
        population, stats, hof, = self._define_ga(toolbox, pop_size)

        population, logbook = self.eaSimpleWithElitism(population,
                                                       toolbox,
                                                       cxpb=cxpb,
                                                       mutpb=mutpb,
                                                       ngen=ngen,
                                                       stats=stats,
                                                       halloffame=hof,
                                                       verbose=vrb)

        best = [self.convert[i] for i in hof.items[0]]
        best_path = ([self.graph.center.index] + best +
                     [self.graph.center.index])
        total_value = self.evaluate_tsp(hof[0])[0]

        if vrb:
            print("-- Best Ever Individual = ", best_path)
            print("-- Best Ever Fitness = ", hof.items[0].fitness.values[0])

        if dir:
            self._plot_ga_results(best_path, logbook, dir, idx)
        else:
            self._plot_ga_results(best_path, logbook).show()

        return best_path, total_value

    def run_ga_vrp(self,
                   agent_num: int,
                   truck_capacity: int,
                   ngen: int = 300,
                   cxpb: float = 0.9,
                   mutpb: float = 0.1,
                   pop_size: int = 500,
                   dir: str | None = None,
                   idx: int = 0,
                   vrb: bool = True) -> tuple[list[int], float]:
        """Runs the Genetic Algorithm for the Vehicle Routing Problem.
        
        This function calls the wrapper functions that define the creator, 
        toolbox and the attributes for the Genetic Algorithm designed to solve
        the Vehicle Routing Problem. It then runs the Genetic Algorithm and 
        returns the best paths found and its total value, while also calling the
        wrapper function to plot the results.

        Args:
            agent_num: The number of agents (trucks).
            truck_capacity: The maximum capacity of a truck.
            ngen (optional): The number of generations. Defaults to 300.
            cxpb (optional): The mating probability. Defaults to 0.9.
            mutpb (optional): The mutation probability. Defaults to 0.1.
            pop_size (optional): The size of the population. Defaults to 500.
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            idx (optional): The index for the plot to save. Defaults to 0.
            vrb: (optional): Run the algorithm in verbose or non-verbose mode.
                Defaults to True.

        Returns:
            A tuple containing the best paths found and its total value.
        """
        self.truck_capacity = truck_capacity
        creator = self._define_creator()
        toolbox = self._define_toolbox_vrp(agent_num)
        population, stats, hof, = self._define_ga(toolbox, pop_size)

        population, logbook = self.eaSimpleWithElitism(population,
                                                       toolbox,
                                                       cxpb=cxpb,
                                                       mutpb=mutpb,
                                                       ngen=ngen,
                                                       stats=stats,
                                                       halloffame=hof,
                                                       verbose=vrb)

        best = hof.items[0]
        zones = self.graph.extract_zones(best)
        best_path = [[self.graph.center.index] + zone +
                     [self.graph.center.index] for zone in zones]
        total_value = self.evaluate_vrp(hof[0])[0]

        if vrb:
            print("-- Best Ever Individual = ", best_path)
            print("-- Best Ever Fitness = ", hof.items[0].fitness.values[0])

        if dir:
            self._plot_ga_results(best, logbook, dir, idx, True)
        else:
            self._plot_ga_results(best, logbook, vrp=True).show()

        return best_path, total_value

    def _plot_ga_results(self,
                         path: list[int],
                         logbook: dict,
                         dir: str | None = None,
                         idx: int = 0,
                         vrp: bool = False) -> plt:
        """Sets up a plotter for the results of the Genetic Algorithm.
        
        This function uses the ``plotter`` module to plot the results of the
        Genetic Algorithm using the ``matplotlib`` library. It creates two
        plots, one showing the map with the path found and the other showing
        the evolution of the best and average fitness values of the population
        across generations.

        Args:
            path: The best path found by the Genetic Algorithm.
            logbook: The logbook containing the statistics of the Genetic
                Algorithm execution.
            dir (optional): The directory where the plots should be saved. Defaults to 
                None, in which case the plot(s) won't be saved.
            idx (optional): The index for the plot to save. Defaults to 0.
            vrp (optional): If the result to plot is for a VRP or a TSP. 
                Defaults to False.

        Returns:
            A ``matplotlib.pyplot`` object containing the plots.
        """
        pltr = plotter.Plotter()
        plt.figure(1)
        if vrp:
            path = [[self.graph.center.index] + z + [self.graph.center.index]
                    for z in self.graph.extract_zones(path)]
            pltr.numOfVehicles = len(path)
        pltr.plot_map(self.graph.create_points(path, vrp=vrp), vrp,
                      self.graph.center.coordinates)
        if dir:
            plt.savefig(f"{dir}/Path{idx}.png")
            plt.clf()
        plt.figure(2)
        pltr.plot_evolution(logbook.select("min"), logbook.select("avg"))
        if dir:
            plt.savefig(f"{dir}/Evolution{idx}.png")
            plt.clf()

        return plt
