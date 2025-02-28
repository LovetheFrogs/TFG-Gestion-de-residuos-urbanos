def instantiate_graph():
    nodes = [
            Node(0, 0, 0, 0, True), 
            Node(1, 1, 1, 2), 
            Node(2, 2, 3, -2), 
            Node(3, 3, 0, 5),
            Node(4, 4, -7, 0)
        ]

    edges = []
    edges.append(Edge(10, self.nodes[0], self.nodes[2]))
    edges.append(Edge(4, self.nodes[1], self.nodes[4]))
    edges.append(Edge(2, self.nodes[3], self.nodes[2]))
    edges.append(Edge(4, self.nodes[2], self.nodes[0]))

    g = Graph()

    for node in self.nodes: self.g.add_node(node)
    for edge in self.edges: self.g.add_edge(edge)