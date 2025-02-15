# Created by LovetheFrogs for URJC's tfg

# Check YAPE (Yet Another Python Formatter)
# Check VSCode LSST docstring plugin
""" TO-DO: Create heuristic to evaluate solutions. ->
-> (min) H(r) = sum(for all nodes) d*x1 + t*x2
 Where:
    r -> path
    d -> distance
    t -> time
    w -> weight of destination (used for edge "value")
    x1, x2, x3 -> weights (adjustable, find optimum...)

    Make edge distances follow this function to penalize/reward?
"""
# TO-DO: Allow to store distance of each node to centroid (for k-clusters algorithm).
# TO-DO: Add non-heuristic search functions (to check if our solution is faster/cheaper). added bfs, left to test. add dijkstra and test both
# TO-DO: Implement first iteration of the algorithm.
# TO-DO: Create benchmark for time to execute & value of different aproaches.
# TO-DO: Add reference to LSST documentation guidelines. -> "https://developer.lsst.io/v/DM-5063/docs/py_docs.html#"
# TO-DO: Create file from database module.
# NOTE: Having a node as visited or not allows for trucks to update the status of various nodes to false to request a new execution of the algorithm, changing the truck that had to visit them.
# RESTRICTION: Central node must be at (0, 0)

""" A module containing the graph definition and functions """

import os
from collections import deque
# from sklearn.cluster import KMeans
import numpy as np
import math
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
        return (abs(self.coordinates[0] - b.coordinates[0]) + abs(self.coordinates[1] - b.coordinates[1]))
        
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
    g.set_center(g.get_node(0))
    res = g.divide_graph(725)
    sg = []
    for z in res: sg.append(g.create_subgraph(z))
    input()
