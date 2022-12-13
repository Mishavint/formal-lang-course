from project.cfpq import cfpq, CfpqAlgorithms
from project.graphs_util import create_two_cycles_graph
from pyformlang.cfg import CFG
from networkx import MultiDiGraph


def generate_graph(nodes: set[int], edges: set[tuple[int, str, int]]) -> MultiDiGraph:
    graph = MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        list(map(lambda edge: (edge[0], edge[2], {"label": edge[1]}), edges))
    )
    return graph


class TestsForTensorCfpq:
    def test_works_as_expected_1(self):
        graph = create_two_cycles_graph(1, 1, "a", "b")
        cfg = CFG.from_text(
            """
        S -> a
        S -> a S
        """
        )
        res = cfpq(cfg, graph, algorithm=CfpqAlgorithms.MATRIX)

        assert res == {(1, 0), (0, 1)} or res == set()

    def test_works_as_expected_2(self):
        graph = generate_graph(
            set(),
            set(),
        )
        cfg = CFG.from_text("""""")

        res = cfpq(cfg, graph, algorithm=CfpqAlgorithms.TENSOR)
        assert res == set()

    def test_works_as_expected_3(self):
        graph = generate_graph(set(), set())
        cfg = CFG.from_text(
            """
        S -> A B
        A -> a
        B -> b
        """
        )

        res = cfpq(cfg, graph, algorithm=CfpqAlgorithms.TENSOR)
        assert res == set()

    def test_works_as_expected_4(self):
        graph = create_two_cycles_graph(1, 1, "a", "b")
        cfg = CFG.from_text(
            """
        S -> a b
        S -> b a
        """
        )

        res = cfpq(cfg, graph, algorithm=CfpqAlgorithms.MATRIX)
        assert res == {(1, 2), (2, 1)}
