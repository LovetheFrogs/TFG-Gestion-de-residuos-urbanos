"""Multiple test cases for the functions coded for the project."""

import unittest
import os
import random
from problem.model import Node, Edge, Graph, load as lg
from utils import create_models as cm
from problem.exceptions import *
from problem.algorithms import Algorithms


class TestNode(unittest.TestCase):
    """Node module tests."""

    def setUp(self):
        self.node = Node(0, 0, 0, 0, True)
        self.nodec = Node(1, 3, 1, 2)

        self.n1 = Node(0, 0, 37.4602, 126.441)
        self.n2 = Node(1, 0, 37.5567, 126.924)

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
        self.assertAlmostEqual(self.n1.get_distance(self.n2), 64.34, delta=0.2)


class TestEdge(unittest.TestCase):
    """Edge module tests."""

    def setUp(self):
        self.node1 = Node(0, 10, 0, 0, True)
        self.node2 = Node(1, 3, 1, 2)

        self.n1 = Node(0, 0, 37.4602, 126.441)
        self.n2 = Node(1, 0, 37.5567, 126.924)

        self.edge = Edge(10, self.node1, self.node2)
        self.edge2 = Edge(50, self.n1, self.n2)

    def test_create(self):
        """Tests the creation of an edge."""
        self.assertAlmostEqual(self.edge2.length, 64.44, delta=0.2)
        self.assertEqual(self.edge.speed, 10)
        self.assertEqual(self.edge.origin, self.node1)
        self.assertEqual(self.edge.dest, self.node2)


class TestGraphCreation(unittest.TestCase):
    """Tests graph creation functions."""

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
            if node.center:
                self.assertEqual(self.g.center, node)
            else:
                self.assertEqual(self.g.node_list[-1], node)
        self.assertEqual(self.g.nodes, 5)
        with self.assertRaisesRegex(DuplicateNode,
                                    "The node is already in the Graph"):
            self.g.add_node(self.nodes[1])

    def test_add_edge(self):
        """ Tests adding a new edge between two nodes of a graph."""
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
            self.assertEqual(self.g.edge_list[-1], edge)
        self.assertEqual(self.g.edges, 4)
        with self.assertRaisesRegex(
                NodeNotFound,
                "The node searched for was not found in the structure. "
                "Index searched: 5"):
            self.g.add_edge(Edge(1, Node(5, 3, 4, 4), self.nodes[0]))
        with self.assertRaisesRegex(
                NodeNotFound,
                "The node searched for was not found in the structure. "
                "Index searched: 8"):
            self.g.add_edge(Edge(1, self.nodes[0], Node(8, 3, 8, 8)))
        with self.assertRaisesRegex(DuplicateEdge,
                                    "The edge is already in the Graph"):
            self.g.add_edge(self.edges[1])

    def test_get_node(self):
        """Tests getting a node from the graph."""
        for node in self.nodes:
            self.g.add_node(node)
        self.assertEqual(self.g.get_node(1), self.nodes[1])
        with self.assertRaisesRegex(
                NodeNotFound,
                "The node searched for was not found in the structure. "
                "Index searched: 10"):
            self.g.get_node(10)

    def test_get_edge(self):
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
        self.assertAlmostEqual(self.g.get_edge(self.nodes[0],
                                               self.nodes[2]).length,
                               555.97,
                               delta=0.2)
        with self.assertRaisesRegex(
                EdgeNotFound,
                "The edge was not found in the structure. Edge 0 -> 1"):
            self.g.get_edge(self.nodes[0], self.nodes[1])

    def test_center(self):
        """Tests if the center node has been assigned properly."""
        for node in self.nodes:
            self.g.add_node(node)
        self.assertTrue(self.g.center is not None)
        self.assertEqual(self.g.center, self.nodes[0])

    def test_from_file(self):
        """Tests generating nodes and edges from a file"""
        self.g.populate_from_file(os.getcwd() + "/tests/files/test.txt")
        self.assertEqual(self.g.get_node(1).index, 1)

    def test_from_TSPLib(self):
        """Tests generating nodes and edges from a TSPLib file."""
        self.g.populate_from_tsplib(os.getcwd() + "/tests/files/berlin52.tsp")
        self.assertEqual(self.g.get_node(1).index, 1)
        self.assertEqual(self.g.nodes, 52)

    def test_distances(self):
        """Test the integrity of the distance matrix."""
        self.g.populate_from_file(os.getcwd() + "/tests/files/test4.txt")
        for i in range(self.g.nodes):
            for j in range(self.g.nodes):
                with self.subTest(i=int(str(i) + str(j))):
                    self.assertEqual(self.g.distances[i][j],
                                     self.g.distances[j][i])

    def test_save_and_load(self):
        """Tests saving and loading a graph."""
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
        self.g.save(f"{os.getcwd()}/problem/data/gbkp")
        aux = lg(f"{os.getcwd()}/problem/data/gbkp")
        self.assertNotEqual(aux, None)
        self.assertTrue(isinstance(aux, Graph))
        self.assertEqual([aux.nodes, aux.edges], [self.g.nodes, self.g.edges])
        os.remove(f"{os.getcwd()}/problem/data/gbkp")


class TestGraphAlgorithms(unittest.TestCase):
    """Tests graph's built-in algorithms."""

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

    def test_bfs(self):
        """Tests bfs."""
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
        self.assertEqual(self.g.bfs(self.nodes[0]), [0, 2])
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[0], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[4]))
        self.g.add_edge(Edge(15, self.nodes[4], self.nodes[3]))
        self.assertEqual(self.g.bfs(self.nodes[0]), [0, 2, 1, 4, 3])

    def test_dijkstra(self):
        """Tests Dijkstra's algorithm."""
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
        self.assertEqual([
            int(v) if v != float('inf') else 300
            for v in list(self.g.dijkstra(0).values())
        ], [300, 695, 300, 300, 0])
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[0], self.nodes[1]))
        self.g.add_edge(Edge(15, self.nodes[2], self.nodes[4]))
        self.g.add_edge(Edge(15, self.nodes[4], self.nodes[3]))
        self.assertEqual([int(v) for v in list(self.g.dijkstra(0).values())],
                         [389, 695, 3752, 2196, 0])

    def test_create_points(self):
        """Tests getting coordinates from a list of node indeces."""
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)
        self.assertEqual(self.g.create_points([0, 2, 3]), [(0.0, 0.0),
                                                           (3.0, -2.0),
                                                           (0.0, 5.0)])


class TestSubgraphCreation(unittest.TestCase):

    def setUp(self):
        self.g = Graph()
        self.g.populate_from_file(f"{os.getcwd()}/tests/files/test2.txt")

    def test_divide_graph(self):
        """Tests graph division into zones."""
        aux = self.g.divide_graph_ascendent(725)
        zones = [[n.index for n in z] for z in aux]
        self.assertEqual(zones,
                         [[0, 9, 3, 4, 10], [0, 11, 2, 7, 8], [0, 5, 6, 1, 12]])

    def test_create_subgraph(self):
        """Tests creating a subgraph from a list of nodes."""
        aux = self.g.divide_graph_ascendent(725)
        for i, zone in enumerate(aux):
            with self.subTest(i=i):
                subgraph = self.g.create_subgraph(zone)
                self.assertEqual(self.g.center.coordinates[0],
                                 subgraph.center.coordinates[0])
                self.assertEqual(self.g.center.coordinates[1],
                                 subgraph.center.coordinates[1])
                self.assertEqual(len(zone), subgraph.nodes)
                self.assertEqual(subgraph.nodes * (subgraph.nodes - 1),
                                 subgraph.edges)
                self.assertEqual(len(subgraph.distances), subgraph.nodes)
                for j in range(subgraph.nodes):
                    for k in range(subgraph.nodes):
                        with self.subTest(i=int(str(j) + str(k))):
                            self.assertEqual(subgraph.distances[j][k],
                                             subgraph.distances[k][j])
                self.assertEqual(subgraph.get_min_num_zones(725), 1)
                self.assertTrue(subgraph.can_pickup_all(725, 1))
                self.assertLess(subgraph.total_weight(), 725)


class TestGraphDefaults(unittest.TestCase):
    """Default graph methods testing"""

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
        for node in self.nodes:
            self.g.add_node(node)
        for edge in self.edges:
            self.g.add_edge(edge)

    def test_length(self):
        """Tests getting the length of a graph object."""
        self.assertEqual(len(self.g), 5)

    def test_get(self):
        """Tests getting an item (node) from a graph."""
        self.assertEqual(self.g[0][0], self.edges[0])
        self.assertEqual(self.g[self.nodes[0]][0], self.edges[0])

    def test_set(self):
        """Tests setting node's edges using []."""
        e1 = Edge(5, self.nodes[4], self.nodes[1])
        self.g[4] = e1
        self.assertEqual(self.g[4], [e1])
        e2 = Edge(6, self.nodes[4], self.nodes[2])
        self.g[4] = e2
        self.assertEqual(self.g[4], [e1, e2])

    def test_contains(self):
        """Tests using contains in a graph."""
        self.assertTrue(self.nodes[0] in self.g)

    def test_bool(self):
        """Tests boolean value of a graph (empty/not empty)."""
        self.assertTrue(self.g)
        g2 = Graph()
        self.assertFalse(g2)

    def test_print(self):
        """Tests result of printing a graph."""
        self.assertEqual(self.g.__repr__(),
                         "Graph with 5 nodes and 4 edges. Center: 0\n")


class TestAlgorithms(unittest.TestCase):
    """Testing of the algorithms used."""

    def setUp(self):
        self.g = Graph()
        self.g.populate_from_file(f"{os.getcwd()}/tests/files/test2.txt")
        self.algo = Algorithms(self.g)
        self.subgraphs = [
            self.g.create_subgraph(z) for z in self.g.divide_graph_ascendent(725)
        ]
        self.g2nodes = Graph()
        self.aux1 = Node(0, 100, 0, 0, True)
        self.aux2 = Node(1, 150, 1, 1)
        self.g2nodes.add_node(self.aux1)
        self.g2nodes.add_node(self.aux2)
        self.g2nodes.add_edge(Edge(1, self.aux1, self.aux2))
        self.g2nodes.set_distance_matrix()
        self.g1node = Graph()
        self.g1node.add_node(self.aux1)
        self.g1node.set_distance_matrix()
        self.g0nodes = Graph()

    def test_ga_tsp(self):
        """Tests the Genetic Algorithm (TSP)"""
        po, vo = self.algo.run_ga_tsp(dir=f"{os.getcwd()}/problem/plots",
                                      name="Path0",
                                      vrb=False)
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        os.remove(f"{os.getcwd()}/problem/plots/Evolution_Path0.png")
        self.assertEqual(po[-1], po[0])
        random_path = random.sample(range(0, self.g.nodes), self.g.nodes)
        self.assertGreater(self.algo._evaluate_tsp(random_path)[0], vo)

        # Test on a graph with two nodes.
        p, v = Algorithms(self.g2nodes).run_ga_tsp(
            dir=f"{os.getcwd()}/problem/plots", name="Path0", vrb=False)
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertTrue(
            p == [self.aux1.index, self.aux2.index, self.aux1.index])

        # Test on a graph with one node.
        p, v = Algorithms(self.g1node).run_ga_tsp(
            dir=f"{os.getcwd()}/problem/plots", name="Path0", vrb=False)
        self.assertEqual(v, 0)
        self.assertTrue(p == [self.aux1.index])

        # Test on a graph with no nodes.
        p, v = Algorithms(self.g0nodes).run_ga_tsp(
            dir=f"{os.getcwd()}/problem/plots", name="Path0", vrb=False)
        self.assertEqual(v, 0)
        self.assertTrue(p == [])

        # Test on a bunch of subgraphs.
        for i in range(len(self.subgraphs)):
            with self.subTest(i=i):
                ps, vs = Algorithms(self.subgraphs[i]).run_ga_tsp(
                    dir=f"{os.getcwd()}/problem/plots", name="Path0", vrb=False)
                os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
                os.remove(f"{os.getcwd()}/problem/plots/Evolution_Path0.png")
                self.assertLessEqual(vs, vo)
                self.assertLessEqual(len(ps), len(po))
                self.assertLess(
                    sum(self.subgraphs[i].get_node(n).weight for n in ps[1:]),
                    725)

    def test_two_opt(self):
        """Tests the 2opt Algorithm."""
        po, vo = self.algo.run_two_opt(dir=f"{os.getcwd()}/problem/plots",
                                       name="Path0")
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        self.assertEqual(po[-1], po[0])
        random_path = random.sample(range(0, self.g.nodes), self.g.nodes)
        self.assertGreater(self.algo.evaluate(random_path), vo)

        # Test on a graph with two nodes.
        p, v = Algorithms(self.g2nodes).run_two_opt(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertTrue(
            p == [self.aux1.index, self.aux2.index, self.aux1.index])

        # Test on a graph with one node.
        p, v = Algorithms(self.g1node).run_two_opt(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [self.aux1.index])

        # Test on a graph with no nodes.
        p, v = Algorithms(self.g0nodes).run_two_opt(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [])

        # Test on a bunch of subgraphs.
        for i in range(len(self.subgraphs)):
            with self.subTest(i=i):
                ps, vs = Algorithms(self.subgraphs[i]).run_two_opt(
                    dir=f"{os.getcwd()}/problem/plots", name="Path0")
                os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
                self.assertLessEqual(vs, vo)
                self.assertLessEqual(len(ps), len(po))
                self.assertLess(
                    sum(self.subgraphs[i].get_node(n).weight for n in ps[1:]),
                    725)

    def test_sa(self):
        """Tests the Simulated Annealing Algorithm."""
        po, vo = self.algo.run_sa(dir=f"{os.getcwd()}/problem/plots",
                                  name="Path0")
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        self.assertEqual(po[-1], po[0])
        random_path = random.sample(range(0, self.g.nodes), self.g.nodes)
        self.assertGreater(self.algo.evaluate(random_path), vo)

        # Test on a graph with two nodes.
        p, v = Algorithms(self.g2nodes).run_sa(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertTrue(
            p == [self.aux1.index, self.aux2.index, self.aux1.index])

        # Test on a graph with one node.
        p, v = Algorithms(self.g1node).run_sa(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [self.aux1.index])

        # Test on a graph with no nodes.
        p, v = Algorithms(self.g0nodes).run_sa(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [])

        # Test on a bunch of subgraphs.
        for i in range(len(self.subgraphs)):
            with self.subTest(i=i):
                ps, vs = Algorithms(self.subgraphs[i]).run_sa(
                    dir=f"{os.getcwd()}/problem/plots", name="Path0")
                os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
                self.assertLessEqual(vs, vo)
                self.assertLessEqual(len(ps), len(po))
                self.assertLess(
                    sum(self.subgraphs[i].get_node(n).weight for n in ps[1:]),
                    725)

    def test_tabu_search(self):
        """Tests the Tabu-serch Algorithm."""
        po, vo = self.algo.run_tabu_search(dir=f"{os.getcwd()}/problem/plots",
                                           name="Path0")
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        self.assertEqual(po[-1], po[0])
        random_path = random.sample(range(0, self.g.nodes), self.g.nodes)
        self.assertGreater(self.algo.evaluate(random_path), vo)

        # Test on a graph with two nodes.
        p, v = Algorithms(self.g2nodes).run_tabu_search(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertTrue(
            p == [self.aux1.index, self.aux2.index, self.aux1.index])

        # Test on a graph with one node.
        p, v = Algorithms(self.g1node).run_tabu_search(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [self.aux1.index])

        # Test on a graph with no nodes.
        p, v = Algorithms(self.g0nodes).run_tabu_search(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [])

        # Test on a bunch of subgraphs.
        for i in range(len(self.subgraphs)):
            with self.subTest(i=i):
                ps, vs = Algorithms(self.subgraphs[i]).run_tabu_search(
                    dir=f"{os.getcwd()}/problem/plots", name="Path0")
                os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
                self.assertLessEqual(vs, vo)
                self.assertLessEqual(len(ps), len(po))
                self.assertLess(
                    sum(self.subgraphs[i].get_node(n).weight for n in ps[1:]),
                    725)


class TestTouringAlgorithms(unittest.TestCase):
    """Tests the tour construction algorithms."""

    def setUp(self):
        self.g = Graph()
        self.g.populate_from_file(f"{os.getcwd()}/tests/files/test2.txt")
        self.algo = Algorithms(self.g)
        self.subgraphs = [
            self.g.create_subgraph(z) for z in self.g.divide_graph_ascendent(725)
        ]
        self.g2nodes = Graph()
        self.aux1 = Node(0, 100, 0, 0, True)
        self.aux2 = Node(1, 150, 1, 1)
        self.g2nodes.add_node(self.aux1)
        self.g2nodes.add_node(self.aux2)
        self.g2nodes.add_edge(Edge(1, self.aux1, self.aux2))
        self.g2nodes.set_distance_matrix()
        self.g1node = Graph()
        self.g1node.add_node(self.aux1)
        self.g1node.set_distance_matrix()
        self.g0nodes = Graph()

    def test_nearest_neighbor(self):
        """Tests the Nearest Neighbor Algorithm."""
        po, vo = self.algo.nearest_neighbor(dir=f"{os.getcwd()}/problem/plots",
                                            name="Path0")
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        self.assertEqual(po[-1], po[0])
        random_path = random.sample(range(0, self.g.nodes), self.g.nodes)
        self.assertGreater(self.algo.evaluate(random_path), vo)

        # Test on a graph with two nodes.
        p, v = Algorithms(self.g2nodes).nearest_neighbor(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertTrue(
            p == [self.aux1.index, self.aux2.index, self.aux1.index])

        # Test on a graph with one node.
        p, v = Algorithms(self.g1node).nearest_neighbor(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [self.aux1.index])

        # Test on a graph with no nodes.
        p, v = Algorithms(self.g0nodes).nearest_neighbor(
            dir=f"{os.getcwd()}/problem/plots", name="Path0")
        self.assertEqual(v, 0)
        self.assertTrue(p == [])

        # Test on a bunch of subgraphs.
        for i in range(len(self.subgraphs)):
            with self.subTest(i=i):
                ps, vs = Algorithms(self.subgraphs[i]).nearest_neighbor(
                    dir=f"{os.getcwd()}/problem/plots", name="Path0")
                os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
                self.assertLessEqual(vs, vo)
                self.assertLessEqual(len(ps), len(po))
                self.assertLess(
                    sum(self.subgraphs[i].get_node(n).weight for n in ps[1:]),
                    725)


class TestLowerBoundAlgorithms(unittest.TestCase):

    def setUp(self):
        self.g = Graph()
        self.g.populate_from_file(f"{os.getcwd()}/tests/files/test2.txt")
        self.algo = Algorithms(self.g)
        self.subgraphs = [
            self.g.create_subgraph(z) for z in self.g.divide_graph_ascendent(725)
        ]
        self.g2nodes = Graph()
        self.aux1 = Node(0, 100, 0, 0, True)
        self.aux2 = Node(1, 150, 1, 1)
        self.g2nodes.add_node(self.aux1)
        self.g2nodes.add_node(self.aux2)
        self.g2nodes.add_edge(Edge(1, self.aux1, self.aux2))
        self.g2nodes.set_distance_matrix()
        self.g1node = Graph()
        self.g1node.add_node(self.aux1)
        self.g1node.set_distance_matrix()
        self.g0nodes = Graph()

    def test_one_tree(self):
        """Tests the 1-tree lower bound Algorithm."""
        e, v = self.algo.one_tree()
        _, vsa = self.algo.run_sa(dir=f"{os.getcwd()}/problem/plots",
                                  name="Path0")
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        self.assertLessEqual(v, vsa)
        with self.assertRaisesRegex(
                NodeNotFound,
                "The node searched for was not found in the structure. "
                "Index searched: 20"):
            self.algo.one_tree(20)
        for i in range(len(e)):
            with self.subTest(i=i):
                edge = e[i]
                self.assertNotEqual(edge[0], -1)
                self.assertNotEqual(edge[1], -1)

        # Test on a graph with two nodes
        e, v = Algorithms(self.g2nodes).one_tree()
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertFalse(e)

        # Test on a graph with one node
        e, v = Algorithms(self.g1node).one_tree()
        self.assertEqual(v, 0)
        self.assertFalse(e)

        # Test on a graph with no nodes
        e, v = Algorithms(self.g0nodes).one_tree()
        self.assertFalse(v)
        self.assertFalse(e)

    def test_held_karp(self):
        """Tests the Held-Karp lower bound algorithm."""
        e, v = self.algo.held_karp_lb()
        _, vsa = self.algo.run_sa(dir=f"{os.getcwd()}/problem/plots",
                                  name="Path0")
        os.remove(f"{os.getcwd()}/problem/plots/Path0.png")
        self.assertLessEqual(v, vsa)
        with self.assertRaisesRegex(
                NodeNotFound,
                "The node searched for was not found in the structure. "
                "Index searched: 20"):
            self.algo.held_karp_lb(20)
        for i in range(len(e)):
            with self.subTest(i=i):
                edge = e[i]
                self.assertNotEqual(edge[0], -1)
                self.assertNotEqual(edge[1], -1)
        self.assertGreaterEqual(v, self.algo.one_tree()[1])

        # Test on a graph with two nodes
        e, v = Algorithms(self.g2nodes).held_karp_lb()
        self.assertEqual(v, self.g2nodes.edge_list[0].value)
        self.assertFalse(e)

        # Test on a graph with one node
        e, v = Algorithms(self.g1node).held_karp_lb()
        self.assertEqual(v, 0)
        self.assertFalse(e)

        # Test on a graph with no nodes
        e, v = Algorithms(self.g0nodes).held_karp_lb()
        self.assertFalse(v)
        self.assertFalse(e)


class TestModelFileCreation(unittest.TestCase):
    """Training file creation script testing"""

    def setUp(self):
        cm.DATA_SIZE = 2
        cm.MIN_NODES = 50
        cm.MAX_NODES = 100

        cm.create_dataset()

    def test_file_creation(self):
        """Tests creating the correct number of files"""
        self.assertTrue(
            os.path.isfile(os.getcwd() + "/utils/datasets/dataset2.txt"))
        self.assertFalse(
            os.path.isfile(os.getcwd() + "/utils/datasets/dataset3.txt"))

    def test_number_of_nodes(self):
        """Tests the number of nodes `n` is between given constraints"""
        for i in range(1, 3):
            with self.subTest(i=i):
                with open(
                        os.getcwd() + "/utils/datasets/dataset" + str(i) +
                        ".txt", "r") as file:
                    n = (int(file.readline().strip()))
                    self.assertTrue(n >= 50 and n <= 100)

    def test_log_creation(self):
        """Tests the log file has been created."""
        self.assertTrue(os.path.isfile(os.getcwd() + "/utils/datasets/log.txt"))

    def tearDown(self):
        os.remove(os.getcwd() + "/utils/datasets/dataset1.txt")
        os.remove(os.getcwd() + "/utils/datasets/dataset2.txt")
        os.remove(os.getcwd() + "/utils/datasets/log.txt")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestAlgorithms('test_simulated_annealing'))
    return suite


def main():
    """Runs all the selected tests."""
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    """Calls main function to run selected tests."""
    main()
