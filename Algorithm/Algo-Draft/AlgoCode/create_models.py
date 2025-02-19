""" Batch creation of randomly generated graph files """

import os
import sys
import random

CWD = os.getcwd()
DATA_SIZE = 20
MIN_NODES, MAX_NODES = 100, 500
MIN_WEIGHT, MAX_WEIGHT = 100, 250
MIN_X, MAX_X = -100, 100
MIN_Y, MAX_Y = -100, 100
MIN_SPEED, MAX_SPEED = 20, 60
NEW_LINE = "\n"

DECORATOR = (
                "+-----------------------------------------"
                "-----------------------------------------+"
            )


def update_global():
    """ Takes script call arguments (if any) and updates the value of the
    extraction constants (nº of nodes, weights, nº of files...)
    """
    if len(sys.argv) % 2 != 1:
        print(
                "Wrong number of arguments, " 
                "please check your call to the module."
             )
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
            case "-x":
                global MIN_X
                MIN_X: int(value)
            case "-X":
                global MAX_X
                MAX_X: int(value)
            case "-y":
                global MIN_Y
                MIN_Y: int(value)
            case "-Y":
                global MAX_Y
                MAX_Y: int(value)
            case "-w":
                global MIN_WEIGHT
                MIN_WEIGHT = float(value)
            case "-W":
                global MAX_WEIGHT
                MAX_WEIGHT = float(value)
            case "-s":
                global MIN_SPEED
                MIN_SPEED = float(value)
            case "-S":
                global MAX_SPEED
                MAX_SPEED = float(value)
            case _:
                continue


def generate_nodes() -> tuple[set[int], int, float, str]:
    """ Generates a set of nodes 
    
    This function generates ``n`` nodes, where ``n`` is between ``MIN_NODES`` 
    and ``MAX_NODES`` and gives them a random weight between ``MIN_WEIGHT`` 
    and ``MAX_WEIGHT`` and random coordinates, while creating the text that 
    will be written to the dataset file.

    Returns:
        A tuple containing a set of the nodes generated, the number of nodes 
        generated, the sum of the weight of all nodes and the node-related text
        to be written to the file.
    """
    num_nodes = random.randint(MIN_NODES, MAX_NODES)
    nodes = set(range(num_nodes))
    node_data = NEW_LINE.join(
        f"{node} {random.uniform(MIN_WEIGHT, MAX_WEIGHT):.2f} " 
        f"{random.randint(MIN_X, MAX_X)} " 
        f"{random.randint(MIN_Y, MAX_Y)}" for node in nodes
    )
    weight = sum(float(line.split()[1]) for line in node_data.split("\n")[0:])
    nodes_data = f"{num_nodes}{NEW_LINE}{node_data}{NEW_LINE}"
    return nodes, len(nodes), weight, nodes_data


def generate_edges(nodes: set[int]) -> tuple[int, str]:
    """ Generates a set of edges

    The function generates ``m`` edges in two iterations. First, it generates
    ``(n - 1) · 2`` edges, two for each node from and to the center to 
    ensure all of them are accesible from the central node. Then, it 
    generates a random set of edges to achieve a density between 0.5 and
    0.75.

    Args
        nodes: A set of all the nodes created.

    Returns
        A tuple containing the number of edges created and the edge-related 
        text to be written to the file.
    """
    nodes_list = list(nodes)
    node_count = len(nodes_list)
    edges = set()
    edge_data = []

    for node in nodes:
        if node == 0: continue
        edge = node, 0
        edges.add(edge)
        edge_data.append(
                            f"{random.uniform(MIN_SPEED, MAX_SPEED):.1f} "
                            f"{node} {0}"
                        )
        edge = 0, node
        edges.add(edge)
        edge_data.append(
                            f"{random.uniform(MIN_SPEED, MAX_SPEED):.1f} "
                            f"{0} {node}"
                        )
    extra_edges = int(
        random.uniform(
            ((0.5 * (node_count * (node_count - 1)))),
            ((0.75 * (node_count * (node_count - 1)))),
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
            f"{random.uniform(MIN_SPEED, MAX_SPEED):.1f} {origin} {dest}"
        )
        edge = (dest, origin)
        edges.add(edge)
        edge_data.append(
            f"{random.uniform(MIN_SPEED, MAX_SPEED):.1f} {dest} {origin}"
        )

    edge_data = f"{len(edges)}{NEW_LINE}" + NEW_LINE.join(edge_data) + NEW_LINE

    return len(edges), edge_data


def create_log(data: list[float], tnodes: int,
               tedges: int, tdensity: float, 
               tweight: float, path: str):
    """ Creates the text to be written to the log.

    Each time the script is run, a log is created where you can see for 
    each dataset generated the number of nodes, edges, total weight and 
    density, as well as the average of these values for the whole datasets.

    Args:
        data: A list containing the information for each dataset (nº of nodes,
            nº of edges, density and total weight).
        tnodes: The total number of nodes for all the datasets.
        tedges: The total number of edges for all the datasets.
        tdensity: The total density for all the datasets.
        tweight: The total weight for all the datasets.
        path: The path where the datasets are saved.
    """
    log_data = (
                f"Generated {DATA_SIZE} datasets.{NEW_LINE}"
                f"{DECORATOR}{NEW_LINE}"
               )
    log_data += NEW_LINE.join(
                                f"dataset{k}{NEW_LINE}    |{NEW_LINE}    |- "
                                f"Nodes: {nodes}{NEW_LINE}    |- "
                                f"Edges: {edges}{NEW_LINE}    |- "
                                f"Density: {density}{NEW_LINE}    "
                                f"|- Weight: {weight}{NEW_LINE}"
                                f"{NEW_LINE}{DECORATOR}"
                                for nodes, edges, density, weight, k in zip(
                                    data[0::5], data[1::5], data[2::5], 
                                    data[3::5], data[4::5]
                                )
                             )
    log_data += (
                    f"{NEW_LINE}Averages{NEW_LINE}    |"f"{NEW_LINE}    "
                    f"|- Nodes: {tnodes / DATA_SIZE}{NEW_LINE}    |- Edges: "
                    f"{tedges / DATA_SIZE}{NEW_LINE}    |- Density: "
                    f"{tdensity / DATA_SIZE}{NEW_LINE}    |- Weight: "
                    f"{tweight / DATA_SIZE}{NEW_LINE}"
                )

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


def main():
    """Calls function to create datasets."""
    create_dataset()


if __name__ == "__main__":
    """Calls main and starts script."""
    main()
