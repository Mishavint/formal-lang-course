import cfpq_data
import networkx
from pathlib import Path


def get_info_from_graph(name: str):
    """
    Prints number of vertices, number of edges and list of labels for graph

    Parameters
    ----------
    name : str
        Name of graph

    Returns
    -------
    tuple
        [num_of_nodes, num_of_edges, (labels)]
    """
    graph = get_graph_by_name(name)

    labels = []
    for edge in graph.edges(data=True):
        labels.append(edge[2]["label"])

    return graph.number_of_nodes(), graph.number_of_edges(), labels


def save_graph_to_dot_file(graph: networkx.Graph, file_path: str):
    """
    Saves graph to chosen file

    Parameters
    ----------
    graph : networkx.Graph
        Graph to save
    file_path : str
        Path to the file

    Raises
    ------
    Exception
        When wrong file exception detected
    """
    file = Path(file_path)

    if file.suffix.lower() != ".dot":
        raise Exception("Only dot extensions")

    pydot_graph = networkx.drawing.nx_pydot.to_pydot(graph)
    pydot_graph.write(file_path)


def create_two_cycles_graph(first_count, second_count, first_label, second_label):
    """
    Parameters
    ----------
    first_count : int
        number of nodes in first cycle
    second_count : int
        number of nodes in second cycle
    first_label : str
    second_label : str

    Returns
    -------
    networkx.MultiDiGraph
        Created graph
    """
    return cfpq_data.labeled_two_cycles_graph(
        first_count, second_count, labels=(first_label, second_label)
    )


def get_graph_by_name(name: str):
    """
    Load graph and returns it

    Parameters
    ----------
    name : str
        Name of graph

    Returns
    -------
    networkx.MultiDiGraph

    Raises
    ----------
    FileNotFoundError
        If graph can not be found by his name

    """
    try:
        path_to_graph = cfpq_data.download(name)
        graph = cfpq_data.graph_from_csv(path_to_graph)
        return graph
    except FileNotFoundError as e:
        raise e


def create_and_save_graph(
    first_count, second_count, first_label, second_label, file_path: str
):
    save_graph_to_dot_file(
        create_two_cycles_graph(
            first_count,
            second_count,
            first_label,
            second_label,
        ),
        file_path,
    )
