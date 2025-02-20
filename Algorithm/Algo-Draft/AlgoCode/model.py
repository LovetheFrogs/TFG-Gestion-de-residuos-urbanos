# Created by LovetheFrogs for URJC's tfg

# Check YAPF (Yet Another Python Formatter)
# Use google code-style - https://google.github.io/styleguide/pyguide.html
# Use google docstrings - https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e
# reStructuredText - https://www.writethedocs.org/guide/writing/reStructuredText/
# PEP 484 - https://www.writethedocs.org/guide/writing/reStructuredText/
# Check https://stackoverflow.com/questions/14328406/tool-to-convert-python-code-to-be-pep8-compliant thread for linters/code style
# TO-DO: Add non-heuristic search functions (to check if our solution is faster/cheaper). added bfs & Dijkstra, left to test both
# TO-DO: Create benchmark for time to execute & value of different aproaches.
# TO-DO: Create file from database module FOR THE OTHER TFG.
# NOTE: Having a node as visited or not allows for trucks to update the status of various nodes to false to request a new execution of the algorithm, changing the truck that had to visit them.
# Investigate 2opt inclusion on GA.
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
import numpy as np
import math
import random
import heapq
import exceptions


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
            f"location: {self.coordinates}. {' Center' if self.center else ''}"
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
        if edge in self.graph: raise DuplicateEdge()
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

    def populate_from_file(self, file: str):
        """Populates a graph from the data in a file.

        An new graph is created form the data available in a file. The file 
        should have a `.txt` extension and the data in it must be formatted 
        accordingly.

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

    def bfs(self, source: Node) -> list[int]:
        """Performs Breadth First Search on the graph from the node ``source``.

        Args:
            source: The index of the start node.

        Returns:
            The path found, represented as a list of the indices of said nodes.

        Raises:
            NodeNotFound: If the start node is not in the Graph.
        """
        if self.get_node(source) not in self.graph:
            raise NodeNotFound(source)
        q = deque()
        snode = self.get_node(source)
        visited = [False] * self.nodes
        visited[source - 1] = True
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

    def dijkstra(self, start: int, end: int) -> list[int]:
        """Performs Dijkstra algorithm on a graph.
        
        Dijkstra is used to find the shortest path between any pair of nodes
        (start, end) in a graph.

        Args:
            start: The index of the starting node.
            end: The index of the final node.

        Returns:
            A list containing the indeces of the nodes that form the path, in 
            the order they should be traversed.
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
            if curr_node == end: break
            if curr_dist > distances[curr_node]: continue

            for edge in self.graph.get(self.get_node(curr_node), []):
                next = edge.dest.index
                new_dist = curr_dist + edge.value
                if new_dist < distances[next]:
                    distances[next] = new_dist
                    childs[next] = curr_node
                    heapq.heappush(pq, (new_dist, next))
        
        return distances[end]

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
            for end in all_nodes:
                key = (start.index, end.index)
                if start != end:
                    print("computing path from", start.index, "to", end.index)
                    distance = self.dijkstra(start.index, end.index)
                    self.shortest_paths[key] = distance
                else: self.shortest_paths[key] = 0

    def create_zones(self, angled_nodes: list[Node], truck_capacity: float) -> list[list[Node]]:
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
                    next_zone.insert(1, zone[1])
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
                    next_zone.insert(1, zone[1])
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
        this can yield not ideal zones (for example, zones that intersect 
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

    def run_GA(self, pop_size=200, ngen=100, cxpb=0.9, mutpb=0.1):
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        print("computing paths")
        self.precompute_shortest_paths()
        print("paths computed")
        nodes = [node.index for node in self.node_list]
        genes = [i for i in range(len(nodes))]
        convert = {i: node for i, node in enumerate(nodes)}

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, genes, len(nodes))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size-1)
            
        def evaluate(individual):
            total_value = 0
            current = self.center
            for ga_idx in individual:
                # Convert GA index to original node index
                original_idx = convert[ga_idx]
                key = (current.index, original_idx)
                total_value += self.shortest_paths.get(key, float('inf'))
                current = self.get_node(original_idx)
            # Return to center
            key = (current.index, self.center.index)
            total_value += self.shortest_paths.get(key, float('inf'))
            # Node order penalty
            penalty = sum(
                self.get_node(convert[ga_idx]).weight * (i + 1)
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
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("local_search", local_search)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)

        algorithms.eaMuPlusLambda(
            pop, toolbox, mu=pop_size, lambda_=2*pop_size,
            cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
            halloffame=hof, verbose=True
        )

        best_individual = [convert[i] for i in hof[0]]
        best_path = [self.center.index] + best_individual + [self.center.index]
        total_value = evaluate(hof[0])[0]

        return best_path, total_value

    def __repr__(self):
        new_line = "\n"
        return f"{new_line.join(f'{node for node in self.graph}')}"
        #return f"{new_line.join(f"{node} = {edges}" for node, edges in self.graph.items())}"


if __name__ == '__main__':
    """An example of creating using the module.
    
    In this example a graph is created from a file, divided in zones & 
    subgraphs and paths are calculated for each subgraph.
    """
    g = Graph()
    #g.populate_from_file(os.getcwd() + "/files/test2.txt")
    g.populate_from_file(os.getcwd() + "/Algorithm/Algo-Draft/AlgoCode/files/test2.txt")
    res = g.divide_graph(725)
    sg = []
    for z in res: sg.append(g.create_subgraph(z))
    print(len(res))
    for z in res: 
        for n in z: print(n.index, end=' ')
        print(f" - {sum(n.weight for n in z)}")
        print()
    p, v = g.run_GA()
    print(f"Path: {p}\nValue: {v}")
    for graph in sg:
        print(graph)
        p, v = graph.run_GA()
        print(f"Path: {p}\nValue: {v}")

