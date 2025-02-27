import unittest
import os
import shutil
import random
from model import Node, Edge, Graph
from model import load as lg
import create_models as cm
from exceptions import *

"""Multiple test cases for the functions coded for the project."""

class TestNode(unittest.TestCase):
    """Node module tests."""
    def setUp(self):
        self.node = Node(0, 0, 0, 0, True)
        self.nodec = Node(1, 3, 1, 2)

    def test_create(self):
        """Tests the creation of a node."""
        self.assertEqual(self.node.index, 0)
        self.assertEqual(self.nodec.weight, 3)

    def test_center(self):
        """Tests if a node has been assigned the `center` value."""
        self.assertFalse(self.nodec.center)
        self.assertTrue(self.node.center)

    def test_change_status(self):
        """Tests if a node changes its visited status"""
        self.assertFalse(self.node.visited)
        self.node.change_status()
        self.assertTrue(self.node.visited)
        self.node.change_status()
        self.assertFalse(self.node.visited)

    def test_distance(self):
        """Tests calculating the distance between two nodes."""
        self.assertEqual(self.node.get_distance(self.nodec), 3)


class TestEdge(unittest.TestCase):
    """Edge module tests."""
    def setUp(self):
        self.node1 = Node(0, 10, 0, 0, True)
        self.node2 = Node(1, 3, 1, 2)
        self.edge = Edge(10, self.node1, self.node2)

    def test_create(self):
        """Tests the creation of an edge."""
        self.assertEqual(self.edge.length, 3)
        self.assertEqual(self.edge.speed, 10)
        self.assertEqual(self.edge.origin, self.node1)
        self.assertEqual(self.edge.dest, self.node2)


class TestGraph(unittest.TestCase):
    """Graph module tests."""
    def setUp(self):
        self.nodes = [
            Node(0, 0, 0, 0, True), 
            Node(1, 1, 1, 2), 
            Node(2, 2, 3, -2), 
            Node(3, 3, 0, 5),
            Node(4, 4, -7, 0)
        ]
        self.edges = []
        self.edges.append(Edge(10, self.nodes[0], self.nodes[2]))
        self.edges.append(Edge(4, self.nodes[1], self.nodes[4]))
        self.edges.append(Edge(2, self.nodes[3], self.nodes[2]))
        self.edges.append(Edge(4, self.nodes[2], self.nodes[0]))

        self.g = Graph()

    def test_add_node(self):
        """Tests adding a new node to the graph."""
        for node in self.nodes: 
            self.g.add_node(node)
            if node.center: self.assertEqual(self.g.center, node)
            else: self.assertEqual(self.g.node_list[-1], node)
        self.assertEqual(self.g.nodes, 5)
        with self.assertRaisesRegex(
            DuplicateNode, 
            "The node is already in the Graph"
        ):
            self.g.add_node(self.nodes[1])

    def test_add_edge(self):
        """ Tests adding a new edge between two nodes of a graph."""
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
            self.assertEqual(self.g.edge_list[-1], edge)
        self.assertEqual(self.g.edges, 4)
        with self.assertRaisesRegex(
            NodeNotFound, 
            "The node searched for was not found in the structure. "
            "Index searched: 4"
        ):
            self.g.add_edge(Edge(1, Node(4, 3, 4, 4), self.nodes[0]))
        with self.assertRaisesRegex(
            NodeNotFound, 
            "The node searched for was not found in the structure. "
            "Index searched: 8"
        ):
            self.g.add_edge(Edge(1, self.nodes[0], Node(8, 3, 8, 8)))
        with self.assertRaisesRegex(
            DuplicateEdge, 
            "The edge is already in the Graph"
        ):
            self.g.add_edge(self.edges[1])

    def test_get_node(self):
        """Tests getting a node from the graph."""
        for node in self.nodes: self.g.add_node(node)
        self.assertEqual(self.g.get_node(1), self.nodes[1])
        with self.assertRaisesRegex(
            NodeNotFound, 
            "The node searched for was not found in the structure. "
            "Index searched: 10"
        ):
            self.g.get_node(10)

    def test_get_edge(self):
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges: self.g.add_edge(edge)
        self.assertEqual(
            self.g.get_edge(self.nodes[0], self.nodes[2]).length, 5
        )
        with self.assertRaisesRegex(
            EdgeNotFound,
            "The edge was not found in the structure. Edge 0 -> 1"
        ):
            self.g.get_edge(self.nodes[0], self.nodes[1])

    def test_center(self):
        """Tests if the center node has been assigned properly."""
        for node in self.nodes: self.g.add_node(node)
        self.assertTrue(self.g.center is not None)
        self.assertEqual(self.g.center, self.nodes[0])

    def test_from_file(self):
        """Tests generating nodes and edges from a file"""
        self.g.populate_from_file(os.getcwd() + "/files/test.txt")
        self.assertEqual(self.g.get_node(1).index, 1)

    def test_bfs(self):
        """Tests bfs."""
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges: self.g.add_edge(edge)
        self.assertEqual(self.g.bfs(self.nodes[0]), [0, 2])
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[0], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[4]))
        self.g.add_edge(Edge(15, self.nodes[4], self.nodes[3]))
        self.assertEqual(self.g.bfs(self.nodes[0]), [0, 2, 1, 4, 3])

    def test_dijkstra(self):
        """Tests Dijkstra's algorithm."""
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges: self.g.add_edge(edge)
        self.assertEqual(
            [int(v) if v != float('inf') else 300 
                for v in list(self.g.dijkstra(0).values())],
            [300, 5, 300, 300, 0]
        )
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[0], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[4]))
        self.g.add_edge(Edge(15, self.nodes[4], self.nodes[3]))
        self.assertEqual(
            [int(v) for v in list(self.g.dijkstra(0).values())],
            [3, 5, 25, 13, 0]
        )

    def test_create_points(self):
        """Tests getting coordinates from a list of node indeces."""
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges: self.g.add_edge(edge)
        self.assertEqual(
            self.g.create_points([0, 2, 3]),
            [(0.0, 0.0), (3.0, -2.0), (0.0, 5.0)]
        )

    def test_divide_graph(self):
        """Tests graph division into zones."""
        g2 = Graph()
        g2.populate_from_file(f"{os.getcwd()}/files/test2.txt")
        self.assertEqual(
            g2.divide_graph(725),
            [[0, 9, 3, 4, 10], [0, 11, 2, 7, 8], [0, 5, 6, 1, 12]]
        )

    def test_create_subgraph(self):
        """Tests creating a subgraph from a list of nodes."""
        g2 = Graph()
        g2.populate_from_file(f"{os.getcwd()}/files/test2.txt")
        aux = g2.divide_graph(725)
        for i, zone in enumerate(aux):
            with self.subTest(i=i):
                subgraph = g2.create_subgraph(zone)
                self.assertEqual(g2.center, subgraph.center)
                self.assertEqual(len(zone), subgraph.nodes)
                self.assertEqual(
                    subgraph.nodes * (subgraph.nodes - 1), subgraph.edges
                )

    def test_ga_tsp(self):
        """Tests the Genetic Algorithm (TSP)"""
        g2 = Graph()
        g2.populate_from_file(f"{os.getcwd()}/files/test2.txt")
        p, v = g2.run_ga_tsp(f"{os.getcwd()}/files/plots")
        shutil.rmtree(os.getcwd() + "/files/plots")
        self.assertTrue(p[-1], p[0])
        self.assertTrue(p[0], 0)
        random_path = (
            [0] + [n for n in random.sample(g2.node_list, g2.nodes - 1)] + [0]
        )
        self.assertTrue(g2.evaluate(random_path)[0], v)

    def test_save_and_load(self):
        """Tests saving and loading a graph."""
        for node in self.nodes: self.g.add_node(node)
        for edge in self.edges: self.g.add_edge(edge)
        self.g.save(f"{os.getcwd()}/files/data/gbkp")
        aux = lg(f"{os.getcwd()}/files/data/gbkp")
        self.assertNotEqual(aux, None)
        self.assertTrue(isinstance(aux, Graph))
        self.assertEqual(
            [aux.nodes, aux.edges], 
            [self.g.nodes, self.g.edges]
        )


class TestModelFileCreation(unittest.TestCase):
    """Training file creation script testing"""
    def setUp(self):
        shutil.rmtree(os.getcwd() + "/files/datasets")

        cm.DATA_SIZE = 2
        cm.MIN_NODES = 50
        cm.MAX_NODES = 100

        cm.create_dataset()

    def test_file_creation(self):
        """Tests creating the correct number of files"""
        self.assertTrue(os.path.isfile(
            os.getcwd() + "/files/datasets/dataset2.txt"
        ))
        self.assertFalse(os.path.isfile(
            os.getcwd() + "/files/datasets/dataset3.txt"
        ))

    def test_number_of_nodes(self):
        """Tests the number of nodes `n` is between given constraints"""
        for i in range(1, 3):
            with self.subTest(i=i):
                with open(
                    os.getcwd() + "/files/datasets/dataset" + str(i) + ".txt", 
                    "r"
                ) as file:
                    n = (int(file.readline().strip()))
                    self.assertTrue(n >= 50 and n <= 100)

    def test_log_creation(self):
        """Tests the log file has been created."""
        self.assertTrue(os.path.isfile(
            os.getcwd() + "/files/datasets/log.txt"
        ))

    def tearDown(self):
        os.remove(os.getcwd() + "/files/datasets/dataset1.txt")
        os.remove(os.getcwd() + "/files/datasets/dataset2.txt")
        os.remove(os.getcwd() + "/files/datasets/log.txt")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGraph('test_dijkstra'))
    suite.addTest(TestModelFileCreation('test_file_creation'))
    return suite

def main():
    """Runs all the tests."""
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    """Calls main function to run selected tests."""
    main()