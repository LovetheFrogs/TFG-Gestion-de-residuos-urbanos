# Created by LovetheFrogs for URJC's tfg

# Check YAPF (Yet Another Python Formatter)
# Use google code-style - https://google.github.io/styleguide/pyguide.html
# Use google docstrings - https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e
# PEP 484 - https://www.writethedocs.org/guide/writing/reStructuredText/
# NOTE: Having a node as visited or not allows for trucks to update the status of various nodes to false to request a new execution of the algorithm, changing the truck that had to visit them. FOR THE OTHER TFG
# Restriction: Currently all trucks need to have the same capacity.

"""Data structures that model the problem and its solution.

The module contains the definitions and methods of the ``Node`` and ``Edge`` 
classes, both of which form a ``Graph`` object.

The methods contained allow for search and division of a route setting and
organizing problem, dividing a graph into similar-weighted zones and
finding paths for a graph that minimize the value of the objective function.
"""

import os
from collections import deque
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import heapq
import pickle
import sys
from typing import Union
import elitisim
from exceptions import *
import plotter

def load(path: str) -> Union['Graph', None]:
    """Loads a graph object from a file.

    This function is extracted from the graph class because it should be used
    to get a whole new instance of one.
    
    Args:
        path: The path to load the graph from.

    Returns:
        The graph object inside the file. If the object returned is not an 
        instance of the ``Graph`` class, the function returns ``None``
    """
    with open(path, 'rb') as backup:
        g = pickle.load(backup)
    return g if isinstance(g, Graph) else None


class Node():
    """Implements the custom Node object.

    A custom Object is used in order to ease the access to the data stored 
    inside a Node, such as the weight of it. The drawback is this makes 
    creating a graph object and populating it a bit more complex, although it 
    can be easily automated (see ``Graph.populate_from_file`` function).

    Args:
        index: The index of a node. User must ensure it is unique.
        weight: Weight of the node.
        x: ``x`` coordinate of the node.
        y: ``y`` coordinate of the node.
        center (optional): If the node is the center one. Defaults to False.
    """
    def __init__(self, index: int, weight: float, 
                x: float, y: float, center: bool = False):
        #: int: Index of the node to be created.
        self.index = int(index)
        #: float: Weight of the node. 0 if the node has center set to True.
        self.weight = float(weight) if not bool(center) else 0
        #: bool: If the node is the center.
        self.center = bool(center)
        self.visited = False
        #: tuple(float, float): The coordinates of the node.
        self.coordinates = (float(x), float(y))
        self.angle = 0.0

    def get_distance(self, b: 'Node') -> float:
        """Computes the Manhattan distance between two nodes.
        
        The Manhattan distance is the choice for this library, as it is closer 
        to the real world distance between two points than euclidean distance.

        Args:
            b: The node to which we want to know the distance to.

        Returns:
            The absolute value of the manhhatan distance.  
        """
        return abs(
            (abs(self.coordinates[0] - b.coordinates[0]) + 
             abs(self.coordinates[1] - b.coordinates[1]))
        )
        
    def change_status(self):
        """Visits or unvisits the a depending on the previous status"""
        self.visited = not self.visited

    def __repr__(self) -> str:
        """Changes the default representation of a node.
        
        When print is called on a node, this method is called so that the 
        output is more readable.

        Returns:
            A string representing the node.

        Examples:
            >>> print(Node(1, 150, 10, 5))
            Node 1 -> Weight: 150, location: (10, 5)
            >>> print(Node(0, 0, 3, -2, True))
            Node 0 -> Weight: 0, location: (3, -2). Center
        """
        msg = (
            f"Node {self.index} -> weight: {self.weight}, "
            f"location: {self.coordinates}. {'Center' if self.center else ''}"
        )
        return msg
        
        
class Edge():
    """Implements the custom Edge object.

    A custom Object is used in order to eaase the access to the data stored 
    inside an Edge, such as the lenght, speed, origin and destination nodes 
    of it. It has a few drawbacks, but the use of a class makes the code 
    easier to follow, read and debug.

    The cost of an edge is defined as the sum of the length and the time it
    takes to traverse it, although in the future length or time could be
    more important over the other. This attribute (``self.value``) allows 
    to change this easily in the future.

    Args:
        speed: The average speed to traverse the Edge.
        origin: The Node object where this instance of Edge will start.
        dest: The Node object where this instance of Edge will end.

    """
    def __init__(self, speed: float, origin: Node, dest: Node):
        #: float: The length of the edge from origin to dest.
        self.length = origin.get_distance(dest)
        #: float: The average speed to traverse the edge.
        self.speed = float(speed)
        #: Node: The origin of the edge.
        self.origin = origin
        #: Node: The destination of the edge.
        self.dest = dest
        #: float: Time it takes to traverse the edge, given a speed and length.
        self.time = (float(self.length)/1000)/self.speed
        #: float: The cost of the edge, as both length and time affect it.
        self.value = self.length + self.time
        
    def __repr__(self) -> str:
        """Changes the default representation of an edge.
        
        When print is called on an edge, this method is called so that the
        output is more readable.
        
        Returns:
            A string representing the edge.

        Examples:
            >>> print(Edge(50, Node(1, 100, 0, 0), Node(2, 120, 10, 13)))
            Edge 1 - 2 -> value: 23.46
        """
        msg = (
            f"Edge {self.origin.index} - {self.dest.index} -> "
            f"value: {self.value}"
        )
        return msg


class Graph():
    """Contains the definition of the structure and the functions used on it.

    The purpose of this class is to have a definition of the Graph data 
    structure, used to create the distribution map of the bins, as well as 
    containing all the methods used for building, updating and processing 
    the data obtained from the trucks and databases. The Graph class is the 
    core of this solution and contains the methods in charge of finding the 
    solution to the path-finding problem presented.

    When a graph is created it will be empty, leaving adding data to it to the
    user, who can either use a file with data or the provided functions to
    manually add all the nodes and edges.

    Attributes:
        graph (dict): Internal representation of the graph.
        node_list (list[Node]): All the Nodes in the graph.
        edge_list (list[Edge]): All the Edge in the graph.
        nodes (int): Number of nodes in the graph.
        edges (int): Number of edges in the graph.
        center (Node): Central node of the graph. The central node is the start
            of every path (the "distribution center"). 
    """
    def __init__(self):
        self.graph = {}
        self.node_list = []
        self.edge_list = []
        self.nodes = 0
        self.edges = 0
        self.center = None
        
    def get_node(self, idx: int) -> Node:
        """Gets a node from the graph.

        Args:
            idx: The id of the node to search.
        
        Returns:
            The node whose index is equal to the ``idx``parameter.

        Raises:
            NodeNotFound: If the node is not in the graph
        """
        for node in self.graph:
            if node.index == idx: return node
        
        raise NodeNotFound(idx)
          
    def get_edge(self, origin: Node, dest: Node) -> Edge:
        """Gets an edge from the graph.
        
        Args:
            origin: The origin node of the edge.
            dest: The destination node of the edge.
        
        Returns:
            The edge with origin and destination equal to the ones in the 
            arguments.

        Raises:
            NodeNotFound: If the origin/dest nodes are not in the graph.
            EdgeNotFound: If the edge is not in the graph.
        """
        if origin not in self.graph: raise NodeNotFound(origin.index)
        if dest not in self.graph: raise NodeNotFound(dest.index)
        for edge in self.graph[origin]:
            if edge.dest == dest: return edge
        
        raise EdgeNotFound(f"{origin.index} -> {dest.index}")

    def add_node(self, node: Node):
        """Adds a node to the graph. 
        
        The function adds a node to the dict that represents the graph data 
        structure, to the list of nodes and increments the count of nodes 
        contained in the structure. If the node is a center one, it sets 
        self.center to node and doesn't add it to the list of nodes.

        Args:
            node: The node to add to the graph.

        Raises:
            DuplicateNode: If the node is already in the graph.        
        """
        if node not in self.graph: 
            self.graph[node] = []
            self.nodes += 1
            if node.center: self.center = node
            else: self.node_list.append(node)
        else: raise DuplicateNode()
    
    def add_edge(self, edge: Edge):
        """Adds an edge to the graph.

        The function adds an edge to the dict that represents the graph data 
        structure, to the list of edges and increments the count of edges 
        contained in the structure.

        Args:
            edge: The edge to add to the graph

        Raises:
            NodeNotFound: If the origin or dest node of the edge is not in the 
                Graph.
            DuplicateEdge: If the edge is already in the Graph.
        """
        if edge in self.edge_list: raise DuplicateEdge()
        if (edge.origin in self.graph and edge.dest in self.graph):
            self.graph[edge.origin].append(edge)
            self.edge_list.append(edge)
            self.edges += 1
        elif (edge.origin not in self.graph):
            raise NodeNotFound(edge.origin.index)
        else: raise NodeNotFound(edge.dest.index)

    def set_center(self, node: Node):
        """Sets the central node of the graph.

        The central node of the graph is the "distribution center" from where
        the trucks start their paths. Note that an intermediate station where
        a truck can unload is logically just a normal node with weight 0, not
        a central node.

        Args:
            node: The central node of the graph.

        Raises:
            NodeNotFound: If the node is not in the graph.
        """
        if node not in self.node_list: raise NodeNotFound(node.index)
        self.center = node
        self.center.weight = 0
        self.center.center = True
        self.node_list.remove(node)

    def total_weight(self) -> float:
        """Calculates the sum weight of all the nodes in the graph.

        Returns:
            The total weight of all the nodes.
        """
        return sum(node.weight for node in self.node_list)

    def can_pickup_all(self, truck_capacity: float, truck_count: int) -> bool:
        """Computes if all the trucks can pick up the bins in one round.

        Args:
            truck_capacity: The capacity of each truck.
            truck_count: The number of available trucks.

        Returns:
            True if the bins can be picked up in one round, False if not.
        """
        total_waste = self.total_weight()
        total_capacity = truck_capacity * truck_count
        return total_waste < total_capacity

    def set_num_zones(self, truck_capacity: float) -> int:
        """Computes the minimum number of zones.
        
        The minimum number of zones is calculated if each truck's capacity 
        is maximized. As such, the program does not guarantee division in 
        this number of zones, as it may not be possible while keeping zone 
        distribution sensible.

        Args:
            truck_capacity: The capacity of each truck.

        Returns:
            The minimum number of zones.
        """
        return math.ceil(self.total_weight() / truck_capacity)

    def save(self, path: str):
        """Saves a graph to a file.
        
        Args:
            path: The path where the current graph will be saved.
        """
        with open(path, 'wb') as backup:
            pickle.dump(self, backup, protocol = -1)

    def populate_from_file(self, file: str):
        """Populates a graph from the data in a file.

        An new graph is created form the data available in a file. The file 
        should have a `.txt` extension and the data in it must be formatted 
        accordingly.
        
        Note that this function makes the node with index 0 the center one.

        The format of the data in the file must have the following format 
        in order to be readable by this function:
        ```
        n
        idx_1 weight_1 x_1 y_1
        idx_2 weight_2 x_2 y_2
                ...
        idx_n weight_n x_n y_n
        m
        speed_1 origin_1 dest_1
        speed_2 origin_2 dest_2
                ...
        speed_m origin_m dest_m
        ```
        Where ``n`` is the number of nodes to be read, followed by the 
        parameters of each node and ``m`` is the number of edges to be 
        read, followed by the parameters of each edge.

        Args:
            file: The path of the file to read from.
        
        Raises:
            FileNotFoundError: If the file to read is not found.
            IOError: If an error is found when reading a file.
            FileNotFoundError: If the file does not exist.
        """
        with open(file, 'r') as f:
            n = int(f.readline().strip())
            for _ in range(n):
                aux = f.readline().strip().split()
                self.add_node(Node(aux[0], aux[1], aux[2], aux[3]))
            
            m = int(f.readline().strip())
            for _ in range(m):
                aux = f.readline().strip().split()
                self.add_edge(Edge(aux[0], 
                            self.get_node(int(aux[1])), 
                            self.get_node(int(aux[2])))
                            )

            self.set_center(self.get_node(0))
            self.center.center = True

    def bfs(self, source: Node) -> list[int]:
        """Performs Breadth First Search on the graph from the node ``source``.

        Note that this implementation visits nodes in the order they appear in
        the parent node's adjadcency list.

        Args:
            source: The index of the start node.

        Returns:
            The path found, represented as a list of the indices of said nodes.

        Raises:
            NodeNotFound: If the start node is not in the Graph.
        """
        if source not in self.graph:
            raise NodeNotFound(source)
        q = deque()
        snode = source
        visited = [False] * self.nodes
        visited[source.index - 1] = True
        q.append(snode)
        path = []
        curr_val = 0
        while q:
            curr = q.popleft()
            path.append(curr.index)
            for edge in self.graph[curr]:
                curr_val = edge.value
                if not visited[edge.dest.index - 1]:
                    visited[edge.dest.index - 1] = True
                    q.append(edge.dest)

        return path

    def dijkstra(self, start: int) -> list[float]:
        """Performs Dijkstra algorithm on a graph.
        
        Dijkstra is used to find the shortest path between a node and every
        other node in a graph. 

        Args:
            start: The index of the starting node.

        Returns:
            The minimum cost of going from start any other node.
        """
        distances = {
                        node.index: float('inf') for node in self.node_list + 
                        [self.center]
                    }
        distances[start] = 0
        childs = {}
        pq = [(0, start)]

        while pq:
            curr_dist, curr_node = heapq.heappop(pq)
            if curr_dist > distances[curr_node]: continue

            for edge in self.graph.get(self.get_node(curr_node), []):
                next = edge.dest.index
                new_dist = curr_dist + edge.value
                if new_dist < distances[next]:
                    distances[next] = new_dist
                    childs[next] = curr_node
                    heapq.heappush(pq, (new_dist, next))
        
        return distances

    def precompute_shortest_paths(self):
        """Precomputes the shortest path between all node pairs in the graph.
        
        The function precomputes the shortest path between all node pairs and 
        sets ``self.shortest_paths`` to a dict containing this information.
        
        This could be performed using Floyd-Warshall's algorithm, which 
        has a complexity of O(N^3). When compared to Dijkstra's complexity of 
        O(NE + N^2log N), it in theory runs better on not-so-dense graphs, 
        where (E < N^2) but given that our density will be of 1, Dijkstra 
        should (in theory) be faster, although the possibility of changing to 
        Floyd-Warshall in the future might be reconsidered.
        """
        self.shortest_paths = {}
        all_nodes = self.node_list + [self.center]
        for start in all_nodes:
            distances = self.dijkstra(start.index)
            for i, n in enumerate(distances):
                self.shortest_paths[start.index, i] = distances[i]

    def create_zones(
        self, 
        angled_nodes: list[Node], 
        truck_capacity: float
    ) -> list[list[Node]]:
        """Divides the graph in zones.

        This function is called by ``divide_graph()``. It is in charge of 
        dividing the nodes into zones using the ordered list of nodes by
        angle, as discussed in the ``divide_graph()``'s docstring.

        The function uses a greedy approach to evaluate if adding a new node 
        to a zone makes that zone's weight greater than the truck's capacity.

        After determining the nodes in each zone, the central node is added 
        as every path must start and end on it. In other words, every zone 
        must contain the central node.

        Args:
            angled_nodes: List of the graph's nodes ordered by the angle they
                form with the central node.
            truck_capacity: The maximum capacity of each truck.

        Returns:
            A list of lists containing the diferent zones created, each one 
            containing a set of ``Node`` instances.
        """
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

    def postprocess_zones(
        self, 
        zones: list[list[Node]], 
        truck_capacity: float
    ) -> list[list[Node]]:
        """Evaluates zones to determine if a frontier node should be moved.
        
        A frontier node is a node that, while being part of a zone, is right 
        next to another one. This means that in every zone there will be two 
        frontier nodes, the first and last ones of said zone. This function
        determines if moving a frontier node to the zone it is adjacent to 
        is feasible in order to reduce the number of nodes in a zone, with 
        the objective of minimizing the final number of zones. It also does 
        this to make zone's weight similar and avoid big differences in them.

        However, in case this function is comparing two zones with a similar 
        (high) number of nodes, it only evaluates if moving a node equilibrates
        the weight of both zones, as it is not going to eliminate a zone (this 
        would only happen if a zone has at most 3 nodes, including the center).

        To avoid increasing the size of small zones and help eliminate them, 
        the function checks if the previous and next zones have more than 3
        nodes (center, first and last). If not, it skips that zone, as it is 
        possible that zone can be eliminated.

        When the zone being evaluated has one or two nodes, the function tries
        to move this node(s) to contiguous zones to eliminate the current zone.

        Args:
            zones: The zones to be evaluated.
            truck_capacity: The maximum capacity of each truck.

        Returns:
            A list of lists containing the diferent zones created, each one 
            containing a set of ``Node`` instances.
        """
        for i, zone in enumerate(zones):
            prev_zone = zones[i - 1]
            next_zone = zones[i + 1] if i + 1 < len(zones) else zones[0]
            
            prev_big = len(prev_zone) > 3
            next_big = len(next_zone) > 3
            zone_big = len(zone) - 1 > 2

            prev_sum = sum(node.weight for node in prev_zone)
            next_sum = sum(node.weight for node in next_zone)

            first_w = zone[1].weight
            last_w = zone[-1].weight

            zone_empty = not (len(zone) - 1) > 0
            # case 1 & 2
            if ((prev_big or next_big) and (zone_big)):
                # Try move first to prev
                if (prev_sum + first_w) < truck_capacity and prev_big:
                    prev_zone.append(zone[1])
                    zone.remove(zone[1])
                # Try move last to next
                if (
                    (next_sum + last_w) < truck_capacity and 
                    zone_big and 
                    next_big
                ):
                    next_zone.insert(1, zone[-1])
                    zone.remove(zone[-1])
            # case 3
            elif (len(zone) - 1) <= 2:
                # Move first to previous and last to next
                if (prev_sum + first_w) < truck_capacity:
                    prev_zone.append(zone[1])
                    zone.remove(zone[1])
                    if len(zone) == 1:
                        zones.remove(zone)
                        zone_empty = True
                        continue
                if (next_sum + last_w) < truck_capacity and (not zone_empty):
                    next_zone.insert(1, zone[-1])
                    zone.remove(zone[-1])
                    if len(zone) == 1:
                        zones.remove(zone)
                        zone_empty = True
                        continue
                # Move first to next and last to previous
                if (prev_sum + last_w) < truck_capacity and (not zone_empty):
                    prev_zone.append(zone[-1])
                    zone.remove(zone[-1])
                    if len(zone) == 1:
                        zones.remove(zone)
                        zone_empty = True
                        continue
                if (next_sum + first_w) < truck_capacity and (not zone_empty):
                    next_zone.insert(1, zone[1])
                    zone.remove(zone[1])
                    if len(zone) == 1:
                        zones.remove(zone)
                        zone_empty = True
                        continue

        return zones

    def divide_graph(self, truck_capacity: float) -> list[list[Node]]:
        """Manages graph division in zones.

        The division in zones tries to give out a result that minimizes the 
        number of zones while making sure each zone can be fully picked up by 
        one truck. It does not ensure the number of zones is the minimum, as 
        this can yield not-so-ideal zones (for example, zones that intersect 
        others and make the distances between two nodes far too big).

        The zones are divided radially, ordering the nodes depending on the 
        angle they form with ``self.center``. This ensures there is no zone 
        overlapping.

        After division, the zones are post-processed. This looks at the created
        zones and checks if moving the first and/or last node of each zone to 
        a contiguous zone helps lower the total number of zones.
        
        Args:
            truck_capacity: The maximum capacity of each truck.

        Returns:
            A list of lists containing the diferent zones created, each one 
            containing a set of ``Node`` instances.

        Raises:
            NoCenterDefined: If the graph does not have a center node.
            EmptyGraph: If the function is called on an empty graph.
        """
        if self.center is None: raise NoCenterDefined()
        if not self.node_list: raise EmptyGraph()
        for node in self.node_list:
            x_coordinates = node.coordinates[0] - self.center.coordinates[0]
            y_coordinates = node.coordinates[1] - self.center.coordinates[1]
            node.angle = math.atan2(y_coordinates, x_coordinates)
        angled_nodes = sorted(self.node_list, key=lambda n: n.angle)
        zones = self.create_zones(angled_nodes, truck_capacity)
        zones = self.postprocess_zones(zones, truck_capacity)
            
        return zones

    def create_points(self, path: list[int]) -> list[tuple[float, float]]:
        """Generates a list of coordinates from a list of node indices.
        
        Args:
            path: The list of indices of the nodes that form the path.

        Returns:
            The list of coordinates.
        """
        res = []
        for idx in path:
            res.append(self.get_node(idx).coordinates)

        return res

    def create_subgraph(self, nodes: list[Node]) -> 'Graph':
        """Creates a new graph from an existing one.

        The new graph will have the nodes from the current graph that are in 
        the ``nodes`` argument, as well as the edges connecting them, 
        effectively making it a subgraph of the current one.

        Allows to consider each zone as it's own individual graph, making it 
        easier to get the optimal path for a zone.

        Args:
            nodes: A list of nodes for the new graph

        Returns:
            A new subgraph that comes from the graph instance this function is 
            called on.

        Raises:
            NodeNotFound: If the node is not in the graph
        """
        g = Graph()
        for node in nodes: g.add_node(node)
        for node in g.graph:
            if node not in self.graph: raise NodeNotFound(node.index)
            edges = self.graph[node]
            for edge in edges:
                if edge.dest in g.graph: g.add_edge(edge)
        
        return g

    def evaluate(self, individual: list[int]) -> tuple[float, ...]:
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
        total_value = 0
        current = self.center
        for idx in individual:
            original_idx = self.convert[idx]
            total_value += self.get_edge(
                current, self.get_node(original_idx)
                                         ).value
            current = self.get_node(original_idx)
        total_value += self.get_edge(current, self.center).value
        penalty = sum(
            self.get_node(self.convert[idx]).weight * (len(individual) - i)
            for i, idx in enumerate(individual)
        )
        return (total_value + 0.2 * penalty,)
    
    def define_creator(self, vrp: bool = False) -> creator:
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
        
        This wrapper function also allows us to define attributes depending on
        which genetic algorithm we are running; the Genetic Algorithm designed 
        to solve the Vehicle Routing Problem (VRP) or the one that solves
        the Traveling Salesman Problem (TSP).

        Args:
            vrp (optional): True if running the VRP genetic algorithm. Defaults
                to False.

        Returns:
            The creator defined for the genetic algorithm.
        """
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not vrp:
            creator.create(
                                "Individual", 
                                list, typecode='i', 
                                fitness=creator.FitnessMin
                           )

        return creator

    def define_toolbox(self, pop_size: int, vrp: bool = False) -> base.Toolbox:
        """Defines a deap toolbox for the genetic algorithms.
        
        The ``deap.base.createor`` module is part of the DEAP framework. It's 
        used as a container for functions, and enables the creation of new
        operators by customizing existing ones. This function extracts the
        ``toolbox`` instantiation from the ``run_ga_tsp`` function so the code
        is easier to read and follow. 
        
        In the ``toolbox`` object is where the functions used by the genetic
        algorithm are defined, such as the evaluation, selection, crossover
        and mutation functions.
        
        This wrapper function also allows us to define functions depending on
        which genetic algorithm we are running; the Genetic Algorithm designed 
        to solve the Vehicle Routing Problem (VRP) or the one that solves
        the Traveling Salesman Problem (TSP).
        
        Args:
            pop_size: The size of the population.
            vrp (optional): True if running the VRP genetic algorithm. Defaults
                to False.

        Returns:
            The toolbox defined for the genetic algorithm.
        """
        nodes = [node.index for node in self.node_list]
        genes = [i for i in range(len(nodes))]
        self.convert = {i: node for i, node in enumerate(nodes)}

        toolbox = base.Toolbox()
        toolbox.register("random_order", random.sample, genes, len(nodes))
        toolbox.register(
                            "individual_creator",
                            tools.initIterate,
                            creator.Individual,
                            toolbox.random_order
                         )
        toolbox.register(
                            "population_creator",
                            tools.initRepeat,
                            list,
                            toolbox.individual_creator
                         )

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, 
                         indpb=1.0/self.nodes)

        return toolbox

    def define_ga_tsp(self, toolbox: base.Toolbox,
                      pop_size: int) -> tuple[list, dict, list]:
        """Defines the attributes for the TSP Generic Algorithm.
        
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

    def plot_ga_results(
        self, 
        path: list[int], 
        logbook: dict, dir: str | None = None,
        idx: int = 0
    ) -> plt:
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

        Returns:
            A ``matplotlib.pyplot`` object containing the plots.
        """
        pltr = plotter.Plotter()
        plt.figure(1)
        pltr.plot_map(self.create_points(path))
        if dir: plt.savefig(f"{dir}/Path{idx}.png")
        plt.figure(2)
        pltr.plot_evolution(logbook.select("min"), logbook.select("avg"))
        if dir: plt.savefig(f"{dir}/Evolution{idx}.png")

        return plt

    def run_ga_tsp(
        self,
        ngen: int = 100, 
        cxpb: float = 0.9, 
        mutpb: float = 0.1, 
        pop_size: int = 200, 
        dir: str | None = None,
        idx: int = 0,
        vrb: bool = True
    ) -> tuple[list[int], float]:
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
        creator = self.define_creator()
        toolbox = self.define_toolbox(pop_size)
        population, stats, hof, = self.define_ga_tsp(toolbox, pop_size)
        
        population, logbook = elitisim.eaSimpleWithElitism(
                                                    population, 
                                                    toolbox, 
                                                    cxpb=cxpb, 
                                                    mutpb=mutpb,
                                                    ngen=ngen, 
                                                    stats=stats, 
                                                    halloffame=hof, 
                                                    verbose=vrb
                                                  )

        best = [self.convert[i] for i in hof.items[0]]
        best_path = [self.center.index] + best + [self.center.index]
        total_value = self.evaluate(hof[0])[0]

        if vrb:
            print("-- Best Ever Individual = ", best_path)
            print("-- Best Ever Fitness = ", hof.items[0].fitness.values[0])

        if dir: self.plot_ga_results(best_path, logbook, dir, idx)
        else: self.plot_ga_results(best_path, logbook).show()

        return best_path, total_value

    def __repr__(self) -> str:
        """Changes the default representation of a graph.

        When print is called on a graph, this method is called so that the 
        output is more readable.

        Returns:
            A string representing the graph.
            
        Examples:
            >>> print(Graph())
            Graph with 0 nodes and 0 edges. Center: None
            >>> print(Graph().populate_from_file("files/test2.txt"))
            Graph with 13 nodes and 156 edges. Center: 0 
        """
        new_line = "\n"
        msg = (
            f"Graph with {self.nodes} nodes and {self.edges} edges. "
            f"Center: {self.center.index}\n"
        )
        return msg


if __name__ == '__main__':
    """An example of using the module.
    
    In this example a graph is created from a file, divided in zones & 
    subgraphs and paths are calculated for each subgraph.
    """
    g = Graph()
    print("Loading graph")
    g.populate_from_file(os.getcwd() + "/files/test2.txt")
    #g.populate_from_file(os.getcwd() + "/Algorithm/Algo-Draft/AlgoCode/files/test2.txt")
    print("Graph loaded")
    print(g)
    res = g.divide_graph(725)
    sg = []
    print(len(res))
    for i, z in enumerate(res): 
        for n in z: print(n.index, end=' ')
        print(f" - {sum(n.weight for n in z)} Zone {i}")
        sg.append(g.create_subgraph(z))
    p, v = g.run_ga_tsp(dir=f"{os.getcwd()}/files/plots")
    print(f"Path: {p}\nValue: {v}")
    """t = 0
    for graph in sg:
        print(graph)
        p, v = graph.run_ga_tsp()
        t += v
        print(f"Path: {p}\nValue: {v}")
    print(f"Total value: {t}")"""
