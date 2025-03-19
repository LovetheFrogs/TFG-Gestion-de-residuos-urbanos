"""Algorithms used to calculate a path in a graph."""

import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
import heapq
import utils.plotter as plotter
from problem.model import Graph, Node
import problem.exceptions as exceptions


class Algorithms():
    """Class containing the diferent path-finding algorithms.
    
    Args:
        graph: The graph object where the paths will be calculated on.
    """

    def __init__(self, graph: 'Graph'):
        self.graph = graph

    # Function that returns the value of a tour.
    def evaluate(self, individual: list[int]) -> float:
        """Calculates the fitness value of a path.

        Args:
            individual: The path to evaluate.

        Returns:
            The fitness of the path.
        """
        total_value = self.graph.distances[individual[-1]][individual[0]]
        for ind1, ind2 in zip(individual[0:-1], individual[1:]):
            total_value += self.graph.distances[ind1][ind2]
        return (total_value)

    # Lower bound algorithms.
    def one_tree(self,
                 start: int | Node = 0) -> tuple[list[tuple[int, int]], float]:
        """Calculates the 1-tree of the graph.

        A 1-tree is a tree that contains only one cicle.

        The 1-tree of a graph can be used as a lower-bound for the value of 
        a TSP path. This implementation calls Prim's algorithm on a graph
        instance and then adds the shortest edge from the start of the MST to 
        create a cycle.

        Args:
            start: The node where the construction of the MST will start. Can
                either be the index of a node or the Node itself. Defaults to
                0.

        Returns:
            A tuple of the value of the 1-tree and a list of all the edges of 
            the 1-tree, represented as tuples of node indices.
        """
        if self.graph.nodes == 2:
            return None, self.graph.distances[0][1]
        elif self.graph.nodes == 1:
            return None, 0
        elif self.graph.nodes == 0:
            return None, None

        if isinstance(start, Node):
            start = start.index
        if start > max([n.index for n in self.graph.node_list]):
            raise exceptions.NodeNotFound(start)

        result, edges = self.graph.prim(start)
        aux = [(i, w) for i, w in enumerate(self.graph.distances[start])]
        aux.sort(key=lambda x: x[1])
        for item in aux[1:]:
            if not (start, item[0]) in edges:
                result += item[1]
                edges.append((start, item[0]))
                break

        return edges, result

    def held_karp_lb(self,
                     start: int | Node = 0,
                     miter: int = 1000) -> tuple[list[tuple[int, int]], float]:
        """Calculates the held-karp lower bound of a TSP tour.

        The Held-Karp lower bound uses 1-trees to calculate a higher lower
        bound than a 1-tree. To do so, it assigns a *penalty* to each edge and
        updates the edge value accordingly. This penalty is the weight of the 
        source and destination nodes of an edge multiplied by the difference
        between the degree of the node and 2.

        The algorithm has the property where we can get the value every 1-tree 
        generated with this updated edge values would have with the normal 
        values. To do so, we just need to calculate the result of the formula
        :math:`v_{1-tree} - 2 * \\sum_{i = 0}^{n}{\\pi_{i}}` where :mat:`\\pi_{i}`
        is the penalty for node i.

        Args:
            start: The start node to calculate the 1-tree. Defaults to 0.
            miter: The number of iterations of the algorithm. Defaults to 1000.

        Returns:
            A tuple containing the Held-Karp lower-bound and the edges that form
            the tree created by the algorithm. 
        """
        if self.graph.nodes == 2:
            return None, self.graph.distances[0][1]
        elif self.graph.nodes == 1:
            return None, 0
        elif self.graph.nodes == 0:
            return None, None

        if isinstance(start, Node):
            start = start.index
        if start > max([n.index for n in self.graph.node_list]):
            raise exceptions.NodeNotFound(start)

        n = self.graph.nodes
        pi = [self.graph.get_node(i).weight for i in range(n)]
        best_lb = -float('inf')
        best = None
        original_dist = [row[:] for row in self.graph.distances]

        for it in range(miter):
            self.graph.distances = [[
                self.graph.distances[i][j] + pi[i] + pi[j] for j in range(n)
            ] for i in range(n)]

            one_tree_edges, one_tree_value = self.one_tree(start)

            degree = [0] * n
            for i, j in one_tree_edges:
                if i:
                    degree[i] += 1
                if j:
                    degree[j] += 1

            subgrad = [d - 2 for d in degree]

            lb = one_tree_value - (2 * sum(pi))
            if lb > best_lb:
                best_lb = lb
                best = one_tree_edges

            if all(d == 2 for d in degree):
                break

            for i in range(n):
                pi[i] = subgrad[i] * self.graph.get_node(i).weight

        self.graph.distances = original_dist
        return best, best_lb

    # Computing a simple tour.
    def nearest_neighbor(self,
                         start: int | Node = 0,
                         dir: str | None = None,
                         name: str = "") -> tuple[list[int], float]:
        """Gets a tour by executing the Nearest Neighbor algorithm.

        The Nearest Neighbor (NN) algorithm creates a tour by visiting the 
        nodes of a graph in order of distance, visitng the closest next. If the
        closest neighbor is already visited, it moves to the next one and so 
        on.

        Args:
            start: The node from where NN will start. Can either be the index 
            of a node or the Node itself. Defaults to 0.
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            name (optional): The name to add to the plots. Defaults to "".

        Returns:
            A tupple containing the tour and its value.
        """
        if isinstance(start, Node):
            start = start.index

        if self.graph.nodes == 2:
            p = [n.index for n in self.graph.graph.keys()]
            p += [p[0]]
            return p, self.evaluate(p)
        elif self.graph.nodes == 1:
            return [n.index for n in self.graph.graph.keys()], 0
        elif self.graph.nodes == 0:
            return [], 0
        
        path = []
        visited = [False] * self.graph.nodes
        pq = []
        heapq.heappush(pq, start)
        while pq:
            u = heapq.heappop(pq)
            visited[u] = True
            path.append(u)
            neighbors = [(n, d) for n, d in enumerate(self.graph.distances[u])]
            neighbors.sort(key=lambda x: x[1])

            for n, _ in neighbors:
                if not visited[n]:
                    heapq.heappush(pq, n)
                    break

        path.append(path[0])

        if dir:
            self._plot_results(path, dir=dir, name=name)
        else:
            self._plot_results(path).show()

        return path, self.evaluate(path)

    # Private helper functions used in heuristic algorithms.
    def _local_search(self, ind: list[int], mi: int = 50) -> list[int]:
        """Tries to improve the fitness of an individual making use of 2opt.

        Args:
            ind: The individual to improve.
            mi: The maximum number of 2opt iterations. Defaults to 50.

        Returns:
            The improved individual.
        """
        improved = True
        best_fitness = self._evaluate_tsp(ind)
        it = 0
        while improved and it < mi:
            improved = False
            for i in range(1, len(ind), 2):
                for j in range(i + 1, len(ind)):
                    if j - i == 1:
                        continue
                    ni = ind.copy()
                    ni[i:j + 1] = ni[i:j + 1][::-1]
                    new_fitness = self._evaluate_tsp(ind)
                    if new_fitness < best_fitness:
                        ind = ni
                        best_fitness = new_fitness
                        improved = True
                        break
                if improved:
                    break
            it += 1
        return ind

    def _evaluate_tsp(self, individual: list[int]) -> tuple[float, ...]:
        """Wrapper that calls the evaluation function.
        
        The algorithm used for this evaluation function is a genetic 
        algorithms and, as such, it tries to minimize/maximize the value of a
        function to find a solution to a problem. In this case, the problem is
        finding the path in a graph that optimizes a series of objectives 
        (minimizes the value of a fitness function). The ``evaluate_tsp`` 
        wrapper calls the evaluation function to check the result of evaluating 
        said objective function for a given path (individual) and returns a 
        tuple, as required by the DEAP library.
        
        Currently, the objective function gives the cost of the path, which is
        the sum of the values of the edges that form the path, made up of how 
        long they are and the theorical time they take to travel trough times
        2.5 to account for real-world time wastes (traffic stops, dense trafic
        and such). It also accounts for a small time of 2 minutes to pick up 
        a node.
        
        Args:
            individual: The path to evaluate.

        Returns:
            A tuple containing the value of the objective function.
        """
        return (self.evaluate(individual)),

    def _clone(self, ind: list[int]) -> list[int]:
        """Overrides DEAP's cloning to improve time & space performance of the
        genetic algorithm. It takes an individual and returs a copy of it.

        Args:
            ind: The individual to clone.

        Returns:
            The cloned individual.
        """
        new_ind = creator.Individual(ind)
        new_ind.fitness.values = ind.fitness.values
        return new_ind

    def _flip(self, path: list[int], i: int, j: int) -> list[int]:
        """Flips a section of a path.

        Given a path, it flips the section between i and j so that path[i] will
        be path[j], path[i + 1] will be path[j - 1] and so on.

        Args:
            path: The path where a section will be flipped.
            i: The start of a section.
            j: The end of a section.

        Returns:
            The result of performing the section flip on the given path.
        """
        new_path = np.concatenate(
            (path[0:i], path[j:-len(path) + i - 1:-1], path[j + 1:len(path)]))
        return [int(n) for n in new_path]

    def _get_neighbors(self, path: list[int]) -> list[list[int]]:
        """Gets all the posible neighbors of a path.

        A neighbor of a path `p` is another path `p'` where the position of
        two nodes has been interchanged using the `_flip` function.

        Args:
            path: The path whose neighbors we want to find.

        Returns:
            The list of neighbor paths.
        """
        neighbors = []
        for i in range(self.graph.nodes):
            for j in range(i + 1, self.graph.nodes):
                n = self._flip(path, i, j)
                neighbors.append(n)
        return neighbors

    def _check_length(self) -> tuple[list[int | None], float] | None:
        """Gives a TSP tour and value for special cases where graphs are small.
         
        This function checks the length of a graph and returns the value of a 
        TSP tour and value in the cases where the graph has 0, q or 2 nodes.

        Returns:
            This function has 4 different returns:
            1- Return a tour with 3 nodes and the length of the single edge
            of the graph when its length is 2.
            2- Return a tour with one node and a value of 0 when the length
            of the graph is 1.
            3- Return an empty list and 0 when the graph is empty.
            4- Return `None` if otherwise.
        """
        if self.graph.nodes == 2:
            if self.graph.center:
                p = [
                    self.graph.center.index, self.graph.node_list[0].index,
                    self.graph.center.index
                ]
            else:
                p = [
                    self.graph.node_list[0].index,
                    self.graph.node_list[1].index,
                    self.graph.node_list[0].index
                ]
            return p, self.graph.distances[0][1]
        elif self.graph.nodes == 1:
            if self.graph.center:
                return [self.graph.center.index], 0
            else:
                return [self.graph.node_list[0]], 0
        elif self.graph.nodes == 0:
            return [], 0
        else:
            return None

    # Private helper functions used in the genetic algorithm.
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

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
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
        toolbox = base.Toolbox()
        toolbox.register("random_order", random.sample,
                         range(self.graph.nodes), self.graph.nodes)
        toolbox.register("individual_creator", tools.initIterate,
                         creator.Individual, toolbox.random_order)
        toolbox.register("population_creator", tools.initRepeat, list,
                         toolbox.individual_creator)

        toolbox.register("evaluate", self._evaluate_tsp)
        toolbox.register("select", tools.selTournament, tournsize=2)

        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate",
                         tools.mutShuffleIndexes,
                         indpb=1.0 / self.graph.nodes)
        toolbox.register("clone", self._clone)

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
        hof = tools.HallOfFame(1)

        return population, stats, hof

    # Metaheuristic algorithms.
    def _eaSimpleWithElitism(self,
                            population,
                            toolbox,
                            cxpb,
                            mutpb,
                            ngen,
                            stats=None,
                            halloffame=None,
                            verbose=__debug__):
        """This algorithm is similar to DEAP eaSimple() algorithm, with the 
        modification that halloffame is used to implement an elitism mechanism. 
        The individuals contained in the halloffame are directly injected into 
        the next generation after aplying a local search optimization and are 
        not subject to the genetic operators of selection, crossover and 
        mutation. The algorithm also adds stagnation detection where, if the 
        algorithm stagnates (does not improve min fitness for a number of 
        generations), the mutation rate increases, or the population is reset 
        using the best individuals.
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

        best_val = float('inf')
        gens_stagnated = 0
        superGens_stagnated = 0
        mut_exploder = 1
        cicles = 0
        mut_exp = min(0.10 * self.graph.nodes, 30)
        stg = min(self.graph.nodes / 1.15, 50)
        mcic = min(self.graph.nodes / 1.15, 25)

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

            elite = halloffame.items
            for i, e in enumerate(elite):
                ie = self._local_search(e)
                e[:] = ie[:]
                e.fitness.values = self._evaluate_tsp(e)

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

            val = halloffame[0].fitness.values[0]

            # Stagnation detection
            if val < best_val:
                best_val = val
                gens_stagnated = 0
                superGens_stagnated = 0
            else:
                gens_stagnated += 1
                superGens_stagnated += 1
            if gens_stagnated >= stg:
                # Mutation rate increase
                if verbose:
                    print("Stagnated")
                if ((mut_exploder < 5 or
                     (mut_exploder < mut_exp and self.graph.nodes >= 20))
                        and mut_exploder < self.graph.nodes):
                    toolbox.register("mutate",
                                     tools.mutShuffleIndexes,
                                     indpb=1 /
                                     (self.graph.nodes - mut_exploder))
                    mut_exploder += 1
                else:
                    if verbose:
                        print("Reseting...")
                    for i, ind in enumerate(population):
                        population[i] = halloffame.items[0]
                    mut_exploder = 1
                    toolbox.register("mutate",
                                     tools.mutShuffleIndexes,
                                     indpb=1 / (self.graph.nodes))
                    cicles += 1
                gens_stagnated = 0

            # Population reseting
            if (cicles >= (mcic) + 1
                    or superGens_stagnated > self.graph.nodes * 6.5):
                if superGens_stagnated > self.graph.nodes * 6.5 and verbose:
                    print("Halted")
                break

        return population, logbook

    def _two_opt(self, path: list[int], vrb: bool) -> tuple[list[int], float]:
        """2-opt algorithm for solving the Travelling Salesman Problem.

        The 2-opt algorithm takes two edges of a path and removes them, it then
        adds two new edges and checking if this improves the value of the 
        path.

        Args:
            path: The starting path from where the 2-opt algorithm is applied.
            threshold: The minimum improvement between 2-opt cicles.
            vrb: Run the algorithm in verbose or non-verbose mode.

        Returns:
            tuple[list[int], float]: _description_
        """
        best = path
        best_value = self.evaluate(best)
        improved = True
        n = self.graph.nodes

        if vrb:
            print(f"Start value: {best_value} - start path: {best}")

        while improved:
            improved = False
            for i in range(0, n - 1):
                for j in range(i + 2, n):
                    prev = self.evaluate(best)
                    curr = self.evaluate(self._flip(best, i, j))

                    if curr < prev:
                        best = self._flip(best, i, j)
                        best_value = self.evaluate(best)
                        improved = True

                    if vrb:
                        print(f"Best value: {best_value} - best path: {best}")

        return best, best_value

    def _simulated_annealing(self, path: list[int], niter: int, mstag: int,
                             vrb: bool) -> tuple[list[int], float]:
        """Simulated Annealing for solving the Travelling Salesman Problem.

        The Simulated Annealing algorithm tries to find a solution by selecting
        a new path, comparing its value to the current path's value and in case
        it is better than the current one or if :math:`p \\in [0, 1]` is less 
        than :math:`e^{-\\frac{\\Delta value}{Temperature}}` it is selected as 
        the current solution.

        Args:
            path: The starting path.
            niter: Maximum number of iterations.
            mstag: Maximum number of iterations without improvements to the 
                value of the objective function.
            vrb: Run the algorithm in verbose or non-verbose mode.

        Returns:
            A tuple containing the best path found and its value.
        """
        current_path = path
        current_value = self.evaluate(current_path)
        best_path = current_path
        best_value = current_value
        temperature = 5000
        alpha = 0.99
        stagnated = 0
        it = 0

        while it < niter and stagnated < mstag:
            i = random.randint(0, self.graph.nodes)
            j = random.randint(0, self.graph.nodes)
            if i > j:
                i, j = j, i
            next_path = self._flip(current_path, i, j)
            next_value = self.evaluate(next_path)

            if ((next_value < current_value)
                    or (random.uniform(0, 1) <= np.exp(
                        (current_value - next_value) / temperature))):
                current_path, current_value = next_path, next_value

                if current_value < best_value:
                    best_path, best_value = current_path, current_value
                    stagnated = 0

            else:
                stagnated += 1
            temperature *= alpha
            it += 1

            if vrb:
                print(f"Iteration {it}. "
                      f"Best value: {best_value} - best path: {best_path} | "
                      f"Temperature: {temperature} | "
                      f"Stagnated: {stagnated}")

        return best_path, best_value

    def _tabu_search(self,
                     path: list[int],
                     niter: int,
                     mstag: int,
                     vrb: bool,
                     tsize: int = 100000) -> tuple[list[int], float]:
        """Tabu-search for solving the Travelling Salesman Problem.

        The Tabu-search algorithm explores a path's neighbors and tries to find
        the best solution aviable by using a tabu list that stores the 
        already-searched for neighbors. It can select a worse solution than
        the current one. This allows to escape local optima.

        Args:
            path: The starting path.
            niter: Maximum number of iterations.
            mstag: Maximum number of iterations without improvements to the 
                value of the objective function.
            tsize (optional): The maximum size of the tabu list. Defaults to 
                100000
            vrb: Run the algorithm in verbose or non-verbose mode.

        Returns:
            A tuple containing the best path found and its value.
        """
        best = path
        current_path = best
        tabu_list = []
        i = 0
        stagnated = 0

        while i < niter and stagnated < mstag:
            neighbors = self._get_neighbors(current_path)
            best_neighbor = None
            best_neighbor_value = float('inf')
            for n in neighbors:
                if n not in tabu_list:
                    neighbor_value = self.evaluate(n)
                    if neighbor_value < best_neighbor_value:
                        best_neighbor = n
                        best_neighbor_value = neighbor_value

            if best_neighbor is None:
                break

            current_path = best_neighbor
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tsize:
                tabu_list.pop(0)

            if self.evaluate(best_neighbor) < self.evaluate(best):
                best = best_neighbor
                stagnated = 0
            else:
                stagnated += 1

            i += 1

            if vrb:
                print(f"Iteration {i}. "
                      f"Best value: {self.evaluate(best)} "
                      f"- best path: {best} | "
                      f"Stagnated: {stagnated}")

        return best, self.evaluate(best)

    # Wrappers to run metaheuristic algorithms.
    def run_ga_tsp(self,
                   ngen: int = 3000,
                   cxpb: float = 0.7,
                   mutpb: float = 0.2,
                   pop_size: int = 1000,
                   dir: str | None = None,
                   name: str = "",
                   vrb: bool = False) -> tuple[list[int], float]:
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
            name (optional): The name to add to the plots. Defaults to "".
            vrb (optional): Run the algorithm in verbose or non-verbose mode.
                Defaults to False.

        Returns:
            A tuple containing the best path found and its total value.
        """
        check = self._check_length()
        if check:
            return check[0], check[1]

        random.seed(169)
        if not self.graph.distances:
            self.graph.set_distance_matrix()

        creator = self._define_creator()
        toolbox = self._define_toolbox()
        population, stats, hof, = self._define_ga(toolbox, pop_size)

        population, logbook = self._eaSimpleWithElitism(population,
                                                       toolbox,
                                                       cxpb=cxpb,
                                                       mutpb=mutpb,
                                                       ngen=ngen,
                                                       stats=stats,
                                                       halloffame=hof,
                                                       verbose=vrb)

        best = [i for i in hof.items[0]]
        best += [best[0]]
        total_value = self._evaluate_tsp(best)[0]

        if vrb:
            print("-- Best Ever Individual = ", best)
            print("-- Best Ever Fitness = ", total_value)

        if dir:
            self._plot_results(best, logbook, dir=dir, name=name)
        else:
            plt = self._plot_results(best, logbook).show()

        return best, total_value

    def run_two_opt(self,
                    path: list[int] | None = None,
                    dir: str | None = None,
                    name: str = "",
                    vrb: bool = False) -> tuple[list[int], float]:
        """Executes 2-opt optimization on a graph.

        Args:
            path (optional): The initial path from which 2-opt is executed. In
                case it is not provided, 2-opt will start on a random path.
                Defaults to None.
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            name (optional): The name to add to the plots. Defaults to "".
            vrb (optional): Run the algorithm in verbose or non-verbose mode.
                Defaults to False.

        Returns:
            A tuple containing the best path found and its total value.
        """
        check = self._check_length()
        if check:
            return check[0], check[1]

        random.seed(169)

        if not path:
            path = random.sample(range(0, self.graph.nodes), self.graph.nodes)
        best, best_value = self._two_opt(path, vrb)
        best += [best[0]]
        
        if dir:
            self._plot_results(best, dir=dir, name=name)
        else:
            self._plot_results(best).show()

        return best, best_value

    def run_sa(self,
               path: list[int] | None = None,
               niter: int = 100000,
               mstag: int = 1500,
               dir: str | None = None,
               name: str = "",
               vrb: bool = False) -> tuple[list[int], float]:
        """Executes Simulated Annealing on a graph

        Args:
            path (optional): The starting path. If it is `None`, a random one
                will be created. Defaults to None.
            niter (optional): Maximum number of iterations. Defaults to 100000.
            mstag (optional): Maximum number of iterations without improvements
                to the value of the objective function. Defaults to 1500.
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            name (optional): The name to add to the plots. Defaults to "".
            vrb (optional): Run the algorithm in verbose or non-verbose mode.
                Defaults to False.

        Returns:
            A tuple containing the best path found and its total value.
        """
        check = self._check_length()
        if check:
            return check[0], check[1]

        random.seed(169)

        if not path:
            path = random.sample(range(0, self.graph.nodes), self.graph.nodes)
        best, best_value = self._simulated_annealing(path,
                                                     niter=niter,
                                                     mstag=mstag,
                                                     vrb=vrb)
        best += [best[0]]
        if dir:
            self._plot_results(best, dir=dir, name=name)
        else:
            self._plot_results(best).show()

        return best, best_value

    def run_tabu_search(self,
                        path: list[int] = None,
                        niter: int = 1000,
                        mstag: int = 100,
                        dir: str | None = None,
                        name: str = "",
                        vrb: bool = False) -> tuple[list[int], float]:
        """Executes Tabu-search on a graph.

            path (optional): The starting path. If it is `None`, a random one
                will be created. Defaults to None.
            niter (optional): Maximum number of iterations. Defaults to 1000.
            mstag (optional): Maximum number of iterations without improvements
                to the value of the objective function. Defaults to 100.
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            name (optional): The name to add to the plots. Defaults to "".
            vrb (optional): Run the algorithm in verbose or non-verbose mode.
                Defaults to False.

        Returns:
            A tuple containing the best path found and its total value.
        """
        check = self._check_length()
        if check:
            return check[0], check[1]

        random.seed(169)

        if not path:
            path = random.sample(range(0, self.graph.nodes), self.graph.nodes)
        best, best_value = self._tabu_search(path,
                                             niter=niter,
                                             mstag=mstag,
                                             vrb=vrb)
        best += [best[0]]
        if dir:
            self._plot_results(best, dir=dir, name=name)
        else:
            self._plot_results(best).show()

        return best, best_value

    # Helper function for plotting results.
    def _plot_results(self,
                      path: list[int],
                      logbook: dict | None = None,
                      dir: str | None = None,
                      name: str = "") -> plt:
        """Sets up a plotter for the results of the Genetic Algorithm.
        
        This function uses the ``plotter`` module to plot the results of the
        Genetic Algorithm using the ``matplotlib`` library. It creates two
        plots, one showing the map with the path found and the other showing
        the evolution of the best and average fitness values of the population
        across generations.

        Args:
            path: The best path found by the Genetic Algorithm.
            logbook (optional): The logbook containing the statistics of the 
                Genetic lgorithm execution. Defaults to None
            dir (optional): The directory where the plots should be saved. 
                Defaults to None, in which case the plot(s) won't be saved.
            name (optional): The name to add to the plots. Defaults to "".

        Returns:
            A ``matplotlib.pyplot`` object containing the plots.
        """
        pltr = plotter.Plotter()
        plt.figure(1)
        pltr.plot_map(self.graph.create_points(path),
                      self.graph.center.coordinates)
        if dir:
            plt.savefig(f"{dir}/{name}.png")
            plt.clf()
        if logbook:
            plt.figure(2)
            pltr.plot_evolution(logbook.select("min"), logbook.select("avg"))
            if dir:
                plt.savefig(f"{dir}/Evolution_{name}.png")
                plt.clf()

        return plt
