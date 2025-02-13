import unittest
import os
import shutil
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
        with self.assertRaisesRegex(DuplicateNode, "The node is already in the Graph"): self.g.add_node(self.nodes[1])

    def test_addEdge(self):
        """ Tests adding a new edge between two nodes of a graph, checking if exceptions are raised when they should """
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
            self.assertEqual(self.g.edge_list[-1], edge)
        self.assertEqual(self.g.edges, 4)
        with self.assertRaisesRegex(NodeNotFound, "The node searched for was not found in the structure. Index searched: 4"):  self.g.add_edge(Edge(1, 1, Node(4, 3), self.nodes[0]))
        with self.assertRaisesRegex(NodeNotFound, "The node searched for was not found in the structure. Index searched: 8"): self.g.add_edge(Edge(1, 1, self.nodes[0], Node(8, 3)))
        with self.assertRaisesRegex(DuplicateEdge, "The edge is already in the Graph"): self.g.add_edge(self.edges[1])

    def test_getNode(self):
        """ Tests getting a node from the graph. Also checks for an exception in case the node does not exist """
        for node in self.nodes: self.g.add_node(node)
        self.assertEqual(self.g.get_node(1), self.nodes[1])
        with self.assertRaisesRegex(NodeNotFound, "The node searched for was not found in the structure. Index searched: 10"): self.g.get_node(10)

    def test_center(self):
        """ Tests if the center node has been assigned properly. """
        for node in self.nodes: self.g.add_node(node)
        self.assertTrue(self.g.center is not None)
        self.assertEqual(self.g.center, self.nodes[len(self.nodes) - 1])

    def test_fromFile(self):
        """ Tests generating nodes and edges from a file """
        self.g.populate_from_file(os.getcwd() + "/files/test.txt")
        self.assertEqual(self.g.get_node(1).index, 1)


class TestModelFileCreation(unittest.TestCase):
    """ Training file creation script testing """
    def setUp(self):
        #shutil.rmtree(os.getcwd() + "/files/datasets")

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
                with open(os.getcwd() + "/files/datasets/dataset" + str(i) + ".txt", "r") as file:
                    n = (int(file.readline().strip()))
                    self.assertTrue(n >= 50 and n <= 100)

    def test_logCreation(self):
        """ Tests the log file has been created """
        self.assertTrue(os.path.isfile(os.getcwd() + "/files/datasets/log.txt"))

    def tearDown(self):
        os.remove(os.getcwd() + "/files/datasets/dataset1.txt")
        os.remove(os.getcwd() + "/files/datasets/dataset2.txt")
        os.remove(os.getcwd() + "/files/datasets/log.txt")



def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestModelFileCreation('setUp'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())