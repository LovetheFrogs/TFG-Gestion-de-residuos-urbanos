""" Batch creation of randomly generated training files """

import os
import sys
import random

CWD = os.getcwd()
DATA_SIZE = 20                      # Number of training files created.
MIN_NODES, MAX_NODES = 500, 1500    # Minimum and Maximum number of nodes.
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

    for k in range(1, DATA_SIZE + 1):
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
        log.append(k)

        with open(f"{path}/dataset{str(k)}.txt", "w") as file:
            file.write(dataset_content)

    create_log(log, tot_nodes, tot_edges, tot_density, tot_weight, path)


if __name__ == "__main__":
    create_dataset()
