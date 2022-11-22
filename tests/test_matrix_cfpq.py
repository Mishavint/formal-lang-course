from project.cfpq import cfpq, CfpqAlgorithms
from project.cfg_util import cfg_from_file
from networkx import MultiDiGraph


def generate_graph(nodes: set[int], edges: set[tuple[int, str, int]]) -> MultiDiGraph:
    graph = MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        list(map(lambda edge: (edge[0], edge[2], {"label": edge[1]}), edges))
    )
    return graph


path = "resources/cfg_files"


class TestsForMatrixCfpq:
    def test_works_as_expected_1(self):
        graph = generate_graph(
            {0, 1, 2, 3, 4, 5},
            {(0, "m", 1), (1, "i", 2), (2, "s", 3), (3, "h", 4), (4, "a", 5)},
        )
        cfg = cfg_from_file(f"{path}/6.txt")
        res = cfpq(cfg, graph, algorithm=CfpqAlgorithms.MATRIX)

        assert res == {(0, 5)}

    def test_works_as_expected_2(self):
        graph = generate_graph(
            {0, 1, 2, 3},
            {(0, "a", 1), (1, "a", 2), (2, "a", 0), (2, "b", 3), (3, "b", 2)},
        )
        cfg = cfg_from_file(f"{path}/7.txt")

        res = cfpq(cfg, graph, start_nodes={3}, algorithm=CfpqAlgorithms.MATRIX)
        assert res == set()

        res = cfpq(cfg, graph, start_nodes={0}, final_nodes={2, 3}, algorithm=CfpqAlgorithms.MATRIX)
        assert res == {(0, 2), (0, 3)}
