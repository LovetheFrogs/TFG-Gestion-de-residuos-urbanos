# Created by LovetheFrogs for URJC's tfg

# Check YAPF (Yet Another Python Formatter)
# Check https://stackoverflow.com/questions/14328406/tool-to-convert-python-code-to-be-pep8-compliant thread for linters/code style
# Check VSCode LSST docstring plugin
# TO-DO: Add non-heuristic search functions (to check if our solution is faster/cheaper). added bfs, left to test. add dijkstra and test both
# TO-DO: Implement first iteration of the algorithm.
# TO-DO: Create benchmark for time to execute & value of different aproaches.
# TO-DO: Create file from database module.
# NOTE: Having a node as visited or not allows for trucks to update the status of various nodes to false to request a new execution of the algorithm, changing the truck that had to visit them.
# Investigate 2opt inclusion on GA.

""" A module containing the graph definition and functions """

import os
from collections import deque
from deap import base, creator, tools, algorithms
# from sklearn.cluster import KMeans
import numpy as np
import math
import random
import heapq
from exceptions import NodeNotFound, DuplicateNode, DuplicateEdge

class Graph():
    """ The Graph class contains the definition of the structure and the functions used on it.

    The purpose of this class is to have a definition of the Graph data structure, used to create the distribution
    map of the bins, as well as to contain all the methods used for building, updating and processing the data
    obtained from the trucks. The Graph class is the core of this solution and contains the methods in charge of
    finding the solution to the path-finding problem presented.

    Methods
    -------
    get_node(idx)
        Returns the node of the graph with the specified id.
    add_node(node)
        Adds a node to the graph.
    add_edge(edge)
        Adds an edge to the graph.
    populate_from_file(file)
        Populates a graph from the data in a file.
    bfs()
        Breadth First Search of the graph.

    Attributes
    ----------
    graph : dict
        The internal representation of the graph data structure. The keys are instances of `Node`
        and the values are instances of `Edge`.
    node_list : list
        A list of all the nodes that make up the graph.
    edge_list : list
        A list of all the edges that make up the graph.
    nodes : int
        The number of nodes in the graph.
    edges : int
        The number of edges in the graph.
    center : Node
        The central node of the graph.

    Raises
    ------
    NodeNotFound
        If a searched for node is not found.
    DuplicateNode
        If a node is already in the Graph.
    FileNotFoundError
        If the file to read is not found.
    IOError
        If an error is found when reading a file.
    """
    def __init__(self):
        self.graph = {}
        self.node_list = []
        self.edge_list = []
        self.nodes = 0
        self.edges = 0
        self.center = None
        
    def get_node(self, idx):
        """ Returns the node of the graph with the specified id.

        Parameters
        ----------
        idx : int
            The id of the node to search.
        
        Returns
        -------
        node : Node
            The node whose index is equal to the ´idx´ parameter.

        Raises
        ------
        NodeNotFound
            If the node is not found
        """
        for node in self.node_list:
            if node.index == idx: return node
        
        raise NodeNotFound(idx)
          
    def get_edge(self, origin, dest):
        if origin not in self.graph: raise NodeNotFound(origin.index)
        if dest not in self.graph: raise NodeNotFound(dest.index)
        for edge in self.graph[origin]:
            if edge.dest == dest: return edge
        
        raise NotImplementedError # Create Exception for when edge is not found

    def add_node(self, node):
        """ Adds a node to the graph. 
        
        The function adds a node to the dict that represents the graph data structure, adding it to the 
        list of nodes and incrementing the count of nodes contained in the structure. If the node is a
        center, it sets self.center to node.

        Parameters
        ----------
        node : Node
            The node to be added to the graph

        Raises
        ------
        DuplicateNode
            If the node is already in the Graph.
        """
        if node not in self.graph: 
            self.graph[node] = []
            self.nodes += 1
            if node.center: self.center = node
            else: self.node_list.append(node)
        else: raise DuplicateNode()
    
    def add_edge(self, edge):
        """ Adds an edge to the graph.

        The function adds an edge to the dict that represents the graph data structure, adding it to the 
        list of edges and incrementing the count of edges contained in the structure.

        Parameters
        ----------
        edge : Edge
            The edge to be added to the graph.

        Raises
        ------
        NodeNotFound
            If the origin node is not in the Graph.
        DuplicateEdge
            If the edge is already in the Graph.
        """
        if edge in self.edge_list: raise DuplicateEdge()
        if (edge.origin in self.graph and edge.dest in self.graph): 
            self.graph[edge.origin].append(edge)
            self.edge_list.append(edge)
            self.edges += 1
        elif (edge.origin not in self.graph): raise NodeNotFound(edge.origin.index)
        else: raise NodeNotFound(edge.dest.index)
        
    def populate_from_file(self, file):
        """ Populates a graph from the data in a file.

        An empty graph is created form the data available in a file. The file should ideally have a `.txt`
        extension. The data in the file should be formatted accordingly. For further information about how
        the file should be formated to be accepted, reffer to the **Notes** section.

        Parameters
        ----------
        file : str
            The path, name and extension of the file to read from as a string.
        
        
        Raises
        ------
        FileNotFoundError
            If the file to read is not found.
        IOError
            If an error is found when reading a file.
        
        See Also
        --------
        graph.Node
        graph.Edge

        Notes
        -----
        The format of the data in the file must have the following format in order to be readable
        by this function:
        ``` text
        n
        idx1 weight1
        idx2 weight2
            ...
        idxn weightn
        m
        length1 speed1 origin1 dest1
        length2 speed2 origin2 dest2
                    ...
        lengthm speedm originm destm
        ```
        Where `n` is the number of nodes to be read, followed by the parameters of each node and 
        `m` is the number of edges to be read, followed by the parameters of each edge.
        """
        with open(file, 'r') as f:
            n = int(f.readline().strip())
            for _ in range(n):
                aux = f.readline().strip().split()
                self.add_node(Node(aux[0], aux[1], aux[2], aux[3]))
            
            m = int(f.readline().strip())
            for _ in range(m):
                aux = f.readline().strip().split()
                self.add_edge(Edge(aux[0], self.get_node(int(aux[1])), self.get_node(int(aux[2]))))

            self.set_center(self.get_node(0))

    def bfs(source):
        """ Performs Breadth First Search on the graph.

        Parameters
        ----------
        source : int
            The index of the start node.

        Returns
        -------
        path : list
            List of the index of the nodes in the order they were visited.
        res : int
            Value of the objective function.

        Raises
        ------
        NodeNotFound
            If the start node is not in the Graph.
        """
        q = deque()
        snode = self.get_node(source)
        visited = [False] * self.nodes
        visited[source - 1] = True
        q.append(snode)
        path = []
        res = 0
        curr_val = 0
        while q:
            res += curr_val
            curr = q.popleft()
            path.append(curr.index)
            for edge in self.graph[curr]:
                curr_val = edge.value
                if not visited[edge.dest.index - 1]:
                    visited[edge.dest.index - 1] = True
                    q.append(edge.dest)

        return path, res

    def dijkstra(self, start, end):
        distances = {node: float('inf') for node in self.node_list + [self.center]}
        distances[start] = 0
        childs = {}
        pq = [(0, start)]

        while pq:
            curr_dist, curr_node = heapq.heappop(pq)
            if curr_node == end: break
            if curr_dist > distances[curr_node]: continue

            for edge in self.graph.get(curr_node, []):
                next = edge.dest
                new_dist = curr_dist + edge.value
                if new_dist < distances[next]:
                    distances[next] = new_dist
                    childs[next] = curr_node
                    heapq.heappush(pq, (new_dist, next))
        
        return distances[end], []

    def precompute_shortest_paths(self):
        self.shortest_paths = {}
        all_nodes = self.node_list + [self.center]
        for start in all_nodes:
            for end in all_nodes:
                key = (start.index, end.index)
                if start != end:
                    distance, _ = self.dijkstra(start, end)
                    self.shortest_paths[key] = distance
                else: self.shortest_paths[key] = 0

    def set_center(self, node):
        self.center = node
        self.center.weight = 0
        self.node_list.remove(node)

    def total_weight(self):
        return sum(node.weight for node in self.node_list)

    def can_pickup_all(self, truck_capacity, truck_count):
        total_waste = self.total_weight()
        total_capacity = truck_capacity * truck_count
        return total_waste < total_capacity

    def set_num_zones(self, truck_capacity):
        return math.ceil(self.total_weight() / truck_capacity)

    def __create_zones__(self, angled_nodes, truck_capacity):
        zones = []
        current_weight = 0
        current_zone = []


        for node in angled_nodes:
            if current_weight + node.weight > truck_capacity:
                zones.append([self.center] + current_zone)
                current_zone = [node]
                current_weight = node.weight
            else:
                current_zone.append(node)
                current_weight += node.weight

        if current_zone: zones.append([self.center] + current_zone)

        return zones

    def __postprocess_zones__(self, zones, truck_capacity):
        # Look at this function further
        improved = True
        while improved:
            improved = False
            i = 0
            while i < len(zones) - 1:
                current_zone = zones[i][1:]
                next_zone = zones[i+1][1:]
                
                if not current_zone:
                    i += 1
                    continue
                
                last_node = current_zone[-1]
                current_weight = sum(n.weight for n in current_zone[:-1])
                new_next_weight = sum(n.weight for n in next_zone) + last_node.weight
                
                if new_next_weight < truck_capacity:
                    zones[i] = [self.center] + current_zone[:-1]
                    zones[i+1] = [self.center] + next_zone + [last_node]
                    if not zones[i][1:]:
                        zones.pop(i)
                        improved = True
                        break
                    improved = True
                i += 1

    def divide_graph(self, truck_capacity, post = False):
        if self.center is None: raise NotImplementedError # Create Exception for when node is not defined
        if not self.node_list: raise NotImplementedError # Create Exception for when there are no nodes added
        for node in self.node_list: node.angle = math.atan2((node.coordinates[1] - self.center.coordinates[1]), (node.coordinates[0] - self.center.coordinates[0]))
        angled_nodes = sorted(self.node_list, key=lambda n: n.angle)
        zones = self.__create_zones__(angled_nodes, truck_capacity)
        if post: self.__postprocess_zones__(zones, truck_capacity)
            
        return zones

    def create_subgraph(self, nodes):
        g = Graph()
        for node in nodes: g.add_node(node)
        for node in g.node_list:
            edges = self.graph[node]
            for edge in edges:
                if edge.dest in g.node_list: g.add_edge(edge)
        
        return g

    def run_GA(self, pop_size=1000, ngen=1000, cxpb=0.6, mutpb=0.4):
        # Reset DEAP classes to avoid conflicts
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual

        # Define FitnessMin with a tuple for weights
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # <-- Tuple here
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.precompute_shortest_paths()
        # Get original non-center node indices (e.g., [1, 2, ..., 12])
        non_center_nodes = [node.index for node in self.node_list]

        # Map GA indices (0-based) to original node indices
        self.ga_to_original = {ga_idx: original_idx for ga_idx, original_idx in enumerate(non_center_nodes)}
        # Map original node indices back to GA indices
        self.original_to_ga = {original_idx: ga_idx for ga_idx, original_idx in self.ga_to_original.items()}

        # Use 0-based GA indices for individuals (e.g., [0, 1, ..., 11])
        non_center_ga_indices = list(self.ga_to_original.keys())

        # DEAP Toolbox setup
        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, non_center_ga_indices, len(non_center_ga_indices))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        def greedy_path():
            """Generates a greedy path using precomputed shortest path costs."""
            start = self.center
            unvisited = self.node_list.copy()
            path = []
            
            while unvisited:
                # Find the next node with the smallest shortest path cost from the current node
                next_node = min(
                    unvisited,
                    key=lambda node: self.shortest_paths.get((start.index, node.index), float('inf'))
                )
                
                # Ensure a valid path exists (graph is connected per problem statement)
                if self.shortest_paths.get((start.index, next_node.index), float('inf')) == float('inf'):
                    raise ValueError(f"No path from {start.index} to {next_node.index}")
                
                # Add the node to the path and mark as visited
                path.append(self.original_to_ga[next_node.index])
                start = next_node
                unvisited.remove(next_node)
            
            return path
        toolbox.register("greedy", greedy_path)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size-1)
            
        def evaluate(individual):
            total_value = 0
            current = self.center
            for ga_idx in individual:
                # Convert GA index to original node index
                original_idx = self.ga_to_original[ga_idx]  # <-- Now valid for 0 ≤ ga_idx ≤ 11
                key = (current.index, original_idx)
                total_value += self.shortest_paths.get(key, float('inf'))
                current = self.get_node(original_idx)
            # Return to center
            key = (current.index, self.center.index)
            total_value += self.shortest_paths.get(key, float('inf'))
            # Node order penalty
            penalty = sum(
                self.get_node(self.ga_to_original[ga_idx]).weight * (i + 1)
                for i, ga_idx in enumerate(individual)
            )
            return (total_value + 0.01 * penalty,)
        
        def local_search(individual):
            improved = True
            while improved:
                improved = False
                for i in range(len(individual)):
                    for j in range(i+2, len(individual)):
                        new_ind = individual[:i] + individual[i:j][::-1] + individual[j:]
                        if evaluate(new_ind)[0] < evaluate(individual)[0]:
                            individual[:] = new_ind
                            improved = True
            return individual,

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("local_search", local_search)

        pop = toolbox.population(n=pop_size) + [creator.Individual(toolbox.greedy())]
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)

        algorithms.eaMuPlusLambda(
            pop, toolbox, mu=pop_size, lambda_=2*pop_size,
            cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
            halloffame=hof, verbose=True
        )

        best_individual = [i + 1 for i in hof[0]]
        best_path = [self.center.index] + best_individual + [self.center.index]
        total_value = evaluate(best_individual[1:-1])[0]

        return best_path, total_value

    def __repr__(self):
        new_line = "\n"
        return f'{new_line.join(f"{node} = {edges}" for node, edges in self.graph.items())}'
        

class Node():
    """ Implements the custom Node object that makes up a Graph.

    A custom Object is used in order to ease the access to the data stored inside a Node, such as the weight of a
    node. The drawback is this makes creating a graph object and populating it a bit more verbose, although it can
    be easily automated.

    Parameters
    ----------
    index : int
        The index of the Node to be created. Must be unique as it is used for search inside a graph, but this
        uniqueness must be provided by the user.
    weight : float
        The weight of the Node to be created. Gets set to 0 is center = True.
    center : bool, optional
        If the node is the central node (source of path). Defaults to false.
    visited : bool
        If the node has been visited. 

    See Also
    --------
    graph.Graph()
    """
    def __init__(self, index, weight, x, y, center = False):
        self.index = int(index)
        self.weight = float(weight) if not bool(center) else 0
        self.center = bool(center)
        self.visited = False
        self.coordinates = (float(x), float(y))
        self.angle = 0.0

    def get_distance(self, b):
        """ Manhattan distance, closer to real world info than euclidean distance"""
        # return math.sqrt(pow(self.coordinates[0] - b.coordinates[0], 2) + pow(self.coordinates[1] - b.coordinates[1], 2))
        return abs((abs(self.coordinates[0] - b.coordinates[0]) + abs(self.coordinates[1] - b.coordinates[1])))
        
    def change_status(self):
        self.visited = not self.visited

    def __repr__(self):
        return f"[ id = {self.index} | weight = {self.weight} | coordinates = {self.coordinates} ]"
        
        
class Edge():
    """ Implements the custom Edge object that makes up a Graph.

    A custom Object is used in order to eaase the access to the data stored inside an Edge, such as the lenght, speed origin
    abd destination of it. It has a few drawbacks, but the use of a class makes the code easier to follow, read and debug.

    Parameters
    ----------
    lenght : float
        The lenght of an Edge.
    speed : float
        The average speed of an Edge.
    origin : Node
        The Node object where this instance of an Edge will start.
    dest : Node
        The Node object where this instance of an Edge will end.

    Attributes
    ----------
    time : float
        The time it takes to go from origin to dest.
    value : float
        The value (cost) of traversing the edge. Comes from the heuristic function,
        making it so it is fixed, while the importance of each part of the objective
        function cahnges its importance. That is, a node with value 100 will have that
        value always, but trough diferent iterations, a lower time may be more important than
        a lower length.

    See Also
    --------
    graph.Node()
    graph.Graph()
    """
    def __init__(self, speed, origin, dest):
        self.length = origin.get_distance(dest)
        self.speed = float(speed)
        self.origin = origin
        self.dest = dest
        self.time = (float(self.length)/1000)/self.speed
        self.value = self.length + self.time
        
    def __repr__(self):
        return f"[ length = {self.length} | speed = {self.speed} | {self.origin.index} -> {self.dest.index} ]"


if __name__ == '__main__':
    g = Graph()
    g.populate_from_file(os.getcwd() + "/files/test2.txt")
    res = g.divide_graph(725)
    sg = []
    for z in res: sg.append(g.create_subgraph(z))
    p, v = g.run_GA()
    print(f"Path: {p}\nValue: {v}")
