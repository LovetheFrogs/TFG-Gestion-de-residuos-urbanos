# :no_entry: [DEPRECATED] -> New documentation located on /reports/model (28.02.2025)
## Algorithm and model 

The following document aims to explain the model created for the proposed problem, as well as the algorithm developed to find a solution, the tests defined for the codebase and other code created within the project.

# Table of contents

agregar guia de estilo o algo así???

1. [Graph model](#1-graph-model)
2. [Algorithm design](#2-algorithm-design)
3. [Testing our model](#3-testing-our-model)
4. [Creating training files](#4-creating-testing-files)
5. [Training the algorithm](#5-training-the-algorithm)
6. [Exceptions](#6-exceptions)
7. [Proving our solution](#7-proving-our-solution)
8. [Requeriments](#8-requeriments)
9. [Footnotes](#9-references)

## 1. Graph model

### Parts of a graph

The problem will be considered as a graph, with the bins being the nodes of thus graph and the paths you can take between any two bins being the edges of the graph. The edges are made of two main things; an **index** unique and used to search for a node with a cleaner, more understandable code and a **weight** wich is the _estimated_ weight a bin will have. This is used in order to plan routes and avoid the trucks being full prematurely.

The edges are quite a bit more complex. Reading the code used to create an instance of an edge:

``` python
def __init__(self, length, speed, origin, dest):
        self.length = length
        self.speed = speed
        self.origin = origin
        self.dest = dest
        self.time = (length/1000)/self.speed
        self.value = self.length + self.time
```

We can see that to create an edge we need the length and average speed of the union between the nodes, and the origin and destination nodes. With his info we get (an estimation of) the time that it takes to take an edge from node _x_ to node _y_; as well as a value (the actual cost of an edge) that is just the sum of the length and time of an edge. This is so edges of the same length can have different values depending on the time they take and vice-versa (more time implies a higher cost, as well as a higher length).

### Defining and creating a graph instance

Having established the two main components of a graph, we can start to take a deeper look into the `Graph` class.

First, lets have a look at its constructor:

```python
def __init__(self):
        self.graph = {}
        self.node_list = []
        self.edge_list = []
        self.nodes = 0
        self.edges = 0
```

A graph instance starts with an empty `dict` (the actual representation of the graph) as well as a bunch of auxiliary values such as the number of nodes and edges and a list of all the nodes and edges. To start populating the graph we can use one of the two options described below:

**1- Manually adding all the desired nodes and edges.**

The `Graph` class has the methods `add_node(node)` and `add_edge(edge)` which, as the name implies, add a node or an edge to the graph. This functions check that both the nodes and edges added are not in the graph already, throwing custom exceptions in case this check fail (see [Exceptions](#exceptions) for more info on this custom exceptions.). For a more detailed explanation of the functions described above, refer to the function's docstring in the module `model.Graph`.

An example of how to use this functions to create a graph is shown in the code snipped below.
```python
def instantiate_graph():
    self.nodes = [Node(i, i) for i in range(5)] # Create 5 instances of Node
    self.edges = []
    self.edges.append(Edge(5, 10, self.nodes[0], self.nodes[2])) # Create 4 Edge instances between some of the nodes we just created
    self.edges.append(Edge(15, 4, self.nodes[1], self.nodes[4]))
    self.edges.append(Edge(8, 2, self.nodes[3], self.nodes[2]))
    self.edges.append(Edge(17, 4, self.nodes[2], self.nodes[0]))

    self.g = Graph() # Instantiate a new empty Graph object

    for node in nodes: g.add_node(node) # Call add_node() on every node we have
    for edge in edges: g.add_edge(edge) # Call add_edge() on every edge we have

    return g
```

**2- Creating a graph from a data file.**

This is the easiest option of both, as the proccess is done automatically. This is done by calling the `populate_from_file(file)` function with a valid path of a properly formated text file. The code used inside automatically creates the nodes and edges needed. The format of the data files must follow the format rules of this example:

```text
n
i1 w1
i2 w2
    ...
in wn
m
l1 s1 o1 d1
l2 s2 o2 d2
    ...
lm sm om dm
```

**Where:**
- $n  : \:number\:of\:nodes\:of\:the\:graph$
- $ix : \:index\:of\:the\:node\:x$
- $wx : \:weight\:of\:the\:node\:x$
- $m  : \:number\:of\:edges\:of\:the\:graph$
- $ly : \:length\:of\:the\:edge\:y$
- $sy : \:average\:speed\:of\:the\:edge\:y$
- $oy, \:dy : \:origin\:and\:destination\:of\:the\:edge\:y$

This format was chosen as it is common in competitive programming when defining graphs, as well as being really easy to read from a function. For more information on how this works, refer to the function's docstrinf in the module `model.Graph`.

### Path-finding inside a graph

The graph model includes various functions to find a "good" path inside an instance. The paths generated by different algorithms are discussed in the [solution proofing](#prooving-our-solution) section, where they are compared against each other to find if our implementation is truly better than a simpler algorithm. The different algorithms are _BFS_[^1], _Dijkstra_[^2] and the custom genetic algorithm used.

## 2. Algorithm design

**TO-DO**

## 3. Testing our model

In order to prove all of our code is working as expected, a `tests.py` module was created. In here, we test all the methods and classes work as intended. This is part of the **[test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development#:~:text=Test%20Driven%20Development%20(TDD)%20is,leading%20to%20more%20robust%20software.)**, where tests are written before the actual functions, making it easier to write working code from the get-go. The only exception for this was the actual algorithm, as the result of the training is undetermined until it is done. Instead of testing the execution of the algorithm as well as obtaining the expected solution, the tests aim to prove the code works, altough a solution cannot be tested for (as oposed to other algorithms like BFS or Dijsktra).

To test the code, the `unittest`[^3] native python module was used, as it can easily be used in any machine with python installed, making portability easier and thus improving the quality of the solution. The tests for a class always follow a similar structure. Each class has it's own test class named like `TestX`, where $x$ is the name of the class to test. For example, to find the tests for the `Graph` class you just have to find the `TestGraph` class inside `tests.py`. In said test classes, the tested for class is tested by comparing the expected result to the actual one. For more complex functions, edge test cases have been used to make sure the code always work.

Altough some default methods have been re-coded for our custom implementation, they have not been tested for as testing, for example, the `__repr__` seems trivial and rather pointless. The `__init__` methods have also not been tested for implicitly, but testing other methods inside each class proves `__init__` works as expected.

Exceptions are tested for in the methods they can raise by forcing them to happen and checking the text of the exception using the `assertRaisesRegex()` function of the module unittest

## 4. Creating training files

To train our genetic algorithm in a wide variety of example graphs that are similar to our map, a helper script `create_models.py` has been created. In this script, several graph files are created, with a random range of nodes, edges and the values of them. By default, the values are as follows:
```python
DATA_SIZE = 20                              # Number of training files created.
MIN_NODES, MAX_NODES = 200, 1000            # Minimum and Maximum number of nodes.
MIN_WEIGHT, MAX_WEIGHT = 100, 1500          # Minimum and Maximum weight of the nodes.
MIN_DIST, MAX_DIST = 0.100, 5.500           # Minimum and Maximum distance between two nodes.
MIN_SPEED, MAX_SPEED = 20, 60               # Minimum and Maximum speed between two nodes.
```

To ensure the training files are somewhat similar to the real world scenario, the graphs created have a **high density**, as in a city, it can be assumed that we can get from a node $x$ to another node $y$ without having to pass by any other node in a majority of node pairs $\{x, y\}$ we can choose. The density of the training graphs will be determined randomly at runtime to be **between 0.50 and 0.60**.

The script runs as a CLI tool and can be called with the command
```shell
cd path/to/script
python3 create_models.py
```
to run with the default values or it can be added some extra arguments to modify those values. Below is a table of all the extra arguments as well as a few examples of command line calls to the script.

<center>

|  Flag  | Description                            |
|--------|:--------------------------------------:|
| **-f** | Number of training files to be created |
| **-n** | Minimum number of nodes                |
| **-N** | Maximum number of nodes                |
| **-w** | Minimum weight of a node               |
| **-W** | Maximum weight of a node               |
| **-d** | Minimum distance between two nodes     |
| **-D** | Maximum distance between two nodes     |
| **-s** | Minimum speed between two nodes        |
| **-S** | Maximum speed between two nodes        |

</center>

**Generate 12 training files**
```shell
python3 create_models.py -f 12
```

**Make a minimum of 10000 nodes and a maximum of 1000000**
```shell
python3 create_models.py -n 10000 -N 1000000
```

**Make the distance between two nodes always be 2.000**
```shell
python3 create_models.py -d 2.000 -D 2.000
```

**Generate only one file, with 150 nodes all separated between 6.000 and 35.000 and a minimum speed of 25**
```shell
python3 create_models.py -f 1 -n 150 -N 150 -d 6.000 -D 35.000 -m 25
```

## 5. Training the algorithm

**TO-DO**

## 6. Exceptions

In order to allow for easier debugging and usage of the software, custom exceptions have been created following python's guidelines, as contrary to other programming languages such as C where the standart practise is to use different return codes for errors in function execution, python used Exceptions that can stop the execution of the program or be catched and treated propeerly. All of the custom exceptions are located in the `exceptions.py` module. 

The exceptions defined are:
- **NodeNotFound:** Raised when a node with index $i$ is searched in a graph and not found. Returns a message and the index searched for.
- **DuplicateNode:** Raised when a node is being added to a graph, but it already exists in that graph instance.

## 7. Proving our solution

In order to prove our solution is better than others found by non-heuristic algorithms, a benchmark module has been created. This module uses the `create_models.py` module to generate 100 models and runs each path-finding algorithm inside the `Graph()` class on each one, getting an average time to find a solution, as well as an average value of the objective function that calculates the cost of using each route.

To create a new benchmark, just execute the module by using the command
```shell
python3 benchmark.py 
```
the result of the benchmark will be in a file called `benchmark_dd_mm_aaaa_hh_mm_ss.txt`, where *dd_mm_aaaa_hh_mm_ss* is the date and time of the benchmark creation. The file will be located inside the `files/bench` folder.

## 8. Requeriments

In order to run the code, you will need the following software:

- **Python >3.10** due to the use of `match` in `create_models.py`

## 9. Footnotes

[^1]: [Breadth First Search](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/)
[^2]: [Dijkstra](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
[^3]: [Unittest documentation](https://docs.python.org/3/library/unittest.html)

## Annex

Module model.py
```python
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
# TO-DO: Add file creation to create_models & refactor it & document it. COMPLETED (run tests). Update `model.md`
# TO-DO: Add check & Exception (& test) for already existing edge. COMPLETED (run tests). Update `model.md`
# TO-DO: Allow to store index of center node (weight 0 for it? visited = None? def center(nodeId) to make a node the center??) in graph AND add visited status to node (T | F). COMPLETED (run tests) Update `model.md`
# TO-DO: Allow to store distance of each node to centroid (for k-clusters algorithm).
# TO-DO: Add non-heuristic search functions (to check if our solution is faster/cheaper). added bfs, left to test. add dijkstra and test both
# TO-DO: Implement first iteration of the algorithm.
# TO-DO: Create benchmark for time to execute & value of different aproaches.
# TO-DO: Add reference to unittest documentation -> "https://docs.python.org/3/library/unittest.html"
# TO-DO: Add reference to LSST documentation guidelines. -> "https://developer.lsst.io/v/DM-5063/docs/py_docs.html#"
# TO-DO: Create file from database module.
# NOTE: Having a node as visited or not allows for trucks to update the status of various nodes to false to request a new execution of the algorithm, changing the truck that had to visit them.

""" A module containing the graph definition and functions """
from collections import deque
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
        
        return NodeNotFound(idx)
        
    
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
            self.node_list.append(node)
            self.nodes += 1
            if node.center: self.center = node
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
        `text
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
        `
        Where `n` is the number of nodes to be read, followed by the parameters of each node and 
        `m` is the number of edges to be read, followed by the parameters of each edge.
        """
        with open(file, 'r') as f:
            n = int(f.readline().strip())
            for _ in range(n):
                aux = f.readline().strip().split()
                self.add_node(Node(aux[0], aux[1]))
            
            m = int(f.readline().strip())
            for _ in range(m):
                aux = f.readline().strip().split()
                self.add_edge(Edge(aux[0], aux[1], self.get_node(aux[2]), self.get_node(aux[3])))

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
    def __init__(self, index, weight, center = False):
        self.index = index
        self.weight = weight if not center else 0
        self.center = center
        self.visited = False
        
    def __repr__(self):
        return f"[ id = {self.index} | weight = {self.weight} ]"
        
        
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
    def __init__(self, length, speed, origin, dest):
        self.length = length
        self.speed = speed
        self.origin = origin
        self.dest = dest
        self.time = (length/1000)/self.speed
        self.value = self.length + self.time
        
    def __repr__(self):
        return f"[ length = {self.length} | speed = {self.speed} | {self.origin.index} -> {self.dest.index} ]"


```

Module exceptions.py
```python
""" Definition of custom exceptions used in the codebase """

class NodeNotFound(Exception):
    """ Used when searching for an object that is not found in the structure
    
    Parameters
    ----------
    message : str, optional
        Description of the error, default is `The node of index {id} was not found in the structure`.
    id : int
        Id of the not found node.
    """
    def __init__(self, id, message = "The node searched for was not found in the structure. Index searched: ", *args):
        self.message = message + str(id)
        self.id = id
        super(NodeNotFound, self).__init__(message, *args)


class DuplicateNode(Exception):
    """ Used if a Node is already part of the graph.
    
    Parameters
    ----------
    message : str, opcional
        Description of the error, default is `The node is already in the Graph`.
    """
    def __init__(self, message = "The node is already in the Graph", *args):
        self.message = message
        super(DuplicateNode, self).__init__(message, *args)


class DuplicateEdge(Exceptio):
    """ Used if an Edge is already part of the graph.

    Parameters
    ----------
    message : str, optional
        Description of the error, default is `The edge is already in the Graph`.
    """
    def __init__(self, message = "The edge is already in the Graph", *args):
        self.message = message
        super(DuplicateEdge, self).__init__(message, *args)

```

Module tests.py
```python
import unittest
import os
from model import Node, Edge, Graph
import create_models as cm
from exceptions import NodeNotFound, DuplicateNode, DuplicateEdge

""" Multiple test cases for the functions coded for the project """

# Test bfs
# Test dijkstra
# Test file-building (file exists et al)

class TestNode(unittest.TestCase):
    """ Node module tests """
    def setUp(self):
        self.node = Node(0, 10)
        self.nodec = Node(1, 3, True)

    def test_create(self):
        """ Tests the creation of a node """
        self.assertEqual(self.node.index, 0)
        self.assertEqual(self.node.weight, 10)

    def test_center(self):
        """ Tests if a node has been assigned the `center` value. """
        self.assertFalse(self.node.center)
        self.assertTrue(self.nodec.center)


class TestEdge(unittest.TestCase):
    """ Edge module tests """
    def setUp(self):
        self.node1 = Node(0, 10)
        self.node2 = Node(1, 5)
        self.edge = Edge(3, 10, self.node1, self.node2)

    def test_create(self):
        """ Tests the creation of an edge """
        self.assertEqual(self.edge.length, 3)
        self.assertEqual(self.edge.speed, 10)
        self.assertEqual(self.edge.origin, self.node1)
        self.assertEqual(self.edge.dest, self.node2)


class TestGraph(unittest.TestCase):
    """ Graph module tests """
    def setUp(self):
        self.nodes = [Node(i, i) for i in range(5)]
        self.nodes.append(Node(5, 13, True))
        self.edges = []
        self.edges.append(Edge(5, 10, self.nodes[0], self.nodes[2]))
        self.edges.append(Edge(15, 4, self.nodes[1], self.nodes[4]))
        self.edges.append(Edge(8, 2, self.nodes[3], self.nodes[2]))
        self.edges.append(Edge(17, 4, self.nodes[2], self.nodes[0]))

        self.g = Graph()

    def test_addNode(self):
        """ Tests adding a new node to the graph, checking if exceptions are raised when they should """
        for node in self.nodes: 
            self.g.add_node(node)
            self.assertEqual(self.g.node_list[-1], node)
        self.assertEqual(self.g.nodes, 6)
        self.assertRaisesRegex(DuplicateNode, "The node is already in the Graph",
                                self.g.add_node(self.nodes[1]))

    def test_addEdge(self):
        """ Tests adding a new edge between two nodes of a graph, checking if exceptions are raised when they should """
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
            self.assertEqual(self.g.edge_list[-1], edge)
        self.assertEqual(self.g.edges, 4)
        self.assertRaisesRegex(NodeNotFound, 
                                "The node searched for was not found in the structure. Index searched: 4",
                                self.g.add_edge(Edge(1, 1, Node(4, 3), self.nodes[0])))
        self.assertRaisesRegex(NodeNotFound, 
                                "The node searched for was not found in the structure. Index searched: 8",
                                self.g.add_edge(Edge(1, 1, self.nodes[0], Node(8, 3))))
        self.assertRaisesRegex(DuplicateEdge,
                                "The edge is already in the Graph",
                                self.g.add_edge(self.edges[1]))

    def test_getNode(self):
        """ Tests getting a node from the graph. Also checks for an exception in case the node does not exist """
        for node in self.nodes: self.g.add_node(node)
        self.assertEquals(self.g.get_node(1), self.nodes[1])
        self.assertRaissesRegex(NodeNotFound,
                                "The node searched for was not found in the structure. Index searched: 10",
                                self.g.get_node(10))

    def test_center(self):
        """ Tests if the center node has been assigned properly. """
        for node in self.nodes: self.g.add_node(node)
        self.assertTrue(self.g.center is not None)
        self.assertEqual(self.g.center is self.nodes[len(self.nodes)])

    def test_fromFile(self):
        """ Tests generating nodes and edges from a file """
        self.g.populate_from_file(os.getcwd() + "/files/test.txt")
        self.assertEqual(self.g.get_node(1).index, 1)


class TestModelFileCreation(unittest.TestCase):
    """ Training file creation script testing """
    def setUp(self):
        cm.DATA_SIZE = 2
        cm.MIN_NODES = 50
        cm.MAX_NODES = 100

        cm.create_dataset()

    def test_fileCreation(self):
        """ Tests creating the correct number of files """
        self.assertTrue(os.path.isfile(os.getcwd() + "/files/datasets/dataset2.txt"))
        self.assertFalse(os.path.isfile(os.getcwd() + "/files/datasets/dataset3.txt"))

    def test_numberOfNodes(self):
        """ Tests the number of nodes `n` is between given constraints """
        for i in range(1, 3):
            with self.subTest(i=i):
                with open(os.getcwd() + "" + str(i) + ".txt", "r") as file:
                    n = (int(file.readline().strip()))
                    self.assertTrue(n >= 50 and n <= 100)

    def test_logCreation(self):
        """ Tests the log file has been created """
        self.assertTrue(os.path.isfile(os.getcwd() + "/files/datasets/log.txt"))

    def tearDown(self):
        os.remove(os.getcwd() + "/files/datasets/dataset1.txt")
        os.remove(os.getcwd() + "/files/datasets/dataset2.txt")

```

Module create_models.py
```python
""" Batch creation of randomly generated training files """

import os
import sys
import random

CWD = os.getcwd()
DATA_SIZE = 20                      # Number of training files created.
MIN_NODES, MAX_NODES = 200, 1000    # Minimum and Maximum number of nodes.
MIN_WEIGHT, MAX_WEIGHT = 100, 1500  # Minimum and Maximum weight of the nodes.
MIN_DIST, MAX_DIST = 0.100, 5.500   # Minimum and Maximum distance between two nodes.
MIN_SPEED, MAX_SPEED = 20, 60       # Minimum and Maximum speed between two nodes.
NEW_LINE = "\n"

DECORATOR = "+---------------------------------------------------------------------------------+"


def update_global():
    """ Takes script call arguments (if any) and updates the value of the
    extraction constants (nº of nodes, weights, nº of files...)
    """
    if len(sys.argv) % 2 != 1:
        print("Wrong number of arguments, please check your call to the module.")
        print("Using default values.")
        return
    for flag, value in zip(sys.argv[1::2], sys.argv[2::2]):
        match flag:
            case "-f":
                global DATA_SIZE
                DATA_SIZE = int(value)
            case "-n":
                global MIN_NODES
                MIN_NODES = int(value)
            case "-N":
                global MAX_NODES
                MAX_NODES = int(value)
            case "-w":
                global MIN_WEIGHT
                MIN_WEIGHT = float(value)
            case "-W":
                global MAX_WEIGHT
                MAX_WEIGHT = float(value)
            case "-d":
                global MIN_DIST
                MIN_DIST = float(value)
            case "-D":
                global MAX_DIST
                MAX_DIST = float(value)
            case "-s":
                global MIN_SPEED
                MIN_SPEED = float(value)
            case "-S":
                global MAX_SPEED
                MAX_SPEED = float(value)
            case _:
                continue


def generate_nodes():
    """ Generates a set of nodes 
    
    This function generates `n` nodes, where n is between MIN_NODES and
    MAX_NODES and gives them a random weight between MIN_WEIGHT and
    MAX_WEIGHT, while creating the text that will be written to the
    dataset file.

    Returns
    -------
    nodes : set()
        A set of the nodes generated.
    len(nodes) : int
        The number of nodes generated `n`.
    weight : float
        The sum of the weight of all nodes.
    nodes_data : str
        The node-related text to be written to the file
    """
    num_nodes = random.randint(MIN_NODES, MAX_NODES)
    nodes = set(range(num_nodes))
    node_data = NEW_LINE.join(
        f"{node} {random.uniform(MIN_WEIGHT, MAX_WEIGHT):.2f}" for node in nodes
    )
    weight = sum(float(line.split()[1]) for line in node_data.split("\n")[0:])
    nodes_data = f"{num_nodes}{NEW_LINE}{node_data}{NEW_LINE}"
    return nodes, len(nodes), weight, nodes_data


def generate_edges(nodes):
    """ Generates a set of edges

    The function generates `m` edges in two iterations. First, it generates
    `n` edges, one for each node to ensure all of them are connected. Then,
    it generates a random set of edges to achieve a density between 0.5 and
    0.75.

    Parameters
    ----------
    nodes : set()
        A set of all the nodes created.

    Returns
    -------
    len(edges) : int
        The number of edges created `m`.
    edge_data : str
        The edge-related text to be written to the file.
    """
    nodes_list = list(nodes)
    node_count = len(nodes_list)
    edges = set()
    edge_data = []

    for node in nodes:
        dest = random.choice(nodes_list)
        while node == dest:
            dest = random.choice(nodes_list)
        edge = (node, dest)
        edge_data.append(
            f"{random.uniform(MIN_DIST, MAX_DIST):.3f} {random.uniform(MIN_SPEED, MAX_SPEED):.1f} {node} {dest}"
        )

    extra_edges = int(
        random.uniform(
            ((0.5 * (node_count * (node_count - 1))) - len(edges)),
            ((0.75 * (node_count * (node_count - 1))) - len(edges)),
        )
    )
    while (len(edges)) < extra_edges:
        origin, dest = random.sample(nodes_list, 2)
        edge = (origin, dest)
        while edge in edges:
            origin, dest = random.sample(nodes_list, 2)
            edge = (origin, dest)
        edges.add(edge)
        edge_data.append(
            f"{random.uniform(MIN_DIST, MAX_DIST):.3f} {random.uniform(MIN_SPEED, MAX_SPEED):.1f} {origin} {dest}"
        )

    edge_data = f"{len(edges)}{NEW_LINE}" + NEW_LINE.join(edge_data) + NEW_LINE

    return len(edges), edge_data


def create_log(data, tnodes, tedges, tdensity, tweight, path):
    """ Creates the text to be written to the log.

    Each time the script is run, a log is created where you can see for each
    dataset generated the number of nodes, edges, total weight and density, as
    well as the average of these values for the whole datasets.

    Parameters
    ----------
    data : list()
        A list containing the information for each dataset (nº of nodes,
        nº of edges, density and total weight).
    tnodes : int
        The total number of nodes for all the datasets.
    tedges: int
        The total number of edges for all the datasets.
    tdensity : int
        The total density for all the datasets.
    tweight : int
        The total weight for all the datasets.
    path : str
        The path where the datasets are saved.
    """
    log_data = f"Generated {DATA_SIZE} datasets.{NEW_LINE}{DECORATOR}{NEW_LINE}"
    log_data += NEW_LINE.join(
        f"dataset{k}{NEW_LINE}    |{NEW_LINE}    |- Nodes: {nodes}{NEW_LINE}    |- Edges: {edges}{NEW_LINE}    |- Density: {density}{NEW_LINE}    |- Weight: {weight}{NEW_LINE}{NEW_LINE}{DECORATOR}"
        for nodes, edges, density, weight, k in zip(
            data[0::5], data[1::5], data[2::5], data[3::5], data[4::5]
        )
    )
    log_data += f"{NEW_LINE}Averages{NEW_LINE}    |{NEW_LINE}    |- Nodes: {tnodes / DATA_SIZE}{NEW_LINE}    |- Edges: {tedges / DATA_SIZE}{NEW_LINE}    |- Density: {tdensity / DATA_SIZE}{NEW_LINE}    |- Weight: {tweight / DATA_SIZE}{NEW_LINE}"
    # Write log
    with open(f"{path}/log.txt", "w") as file:
        file.write(log_data)


def create_dataset():
    """ Function that creates the dataset, log and updates the value of
    the constants.

    This function calls all other functions in this script to create the
    node and edge data, combining them and saving them to a file for each
    dataset. It also updates the constant values if the script is called
    with arguments and generates the log.
    """
    if len(sys.argv) > 1:
        update_global()
    path = CWD + "/files/datasets"
    if not os.path.isdir(path):
        os.makedirs(path)

    tot_nodes, tot_edges, tot_density, tot_weight = 0, 0, 0, 0
    log = []

    for k in range(DATA_SIZE):
        nodes, node_count, weight, node_data = generate_nodes()
        edge_count, edges_data = generate_edges(nodes)
        dataset_content = node_data + edges_data

        tot_nodes += node_count
        tot_edges += edge_count
        tot_density += edge_count / (node_count * (node_count - 1))
        tot_weight += weight

        log.append(node_count)
        log.append(edge_count)
        log.append(edge_count / (node_count * (node_count - 1)))
        log.append(weight)
        log.append(k + 1)

        with open(f"{path}/dataset{str(k)}.txt", "w") as file:
            file.write(dataset_content)

    create_log(log, tot_nodes, tot_edges, tot_density, tot_weight, path)


if __name__ == "__main__":
    create_dataset()

```
