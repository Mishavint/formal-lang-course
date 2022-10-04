from networkx import MultiDiGraph
from project.rpq import rpq, bfs_rpq
from pyformlang.regular_expression import Regex


class TestsForRpq:
    def test_works_as_expected(self):
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "M"}),
                (1, 2, {"label": "i"}),
                (2, 3, {"label": "s"}),
                (3, 4, {"label": "h"}),
                (4, 5, {"label": "a"}),
            ]
        )
        regex = Regex("M i s h a")
        assert rpq(regex, graph) == {(0, 5)}

    def test_empty_graph(self):
        graph = MultiDiGraph()
        assert rpq(Regex("Not empty regex"), graph) == set()

    def test_empty_query(self):
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "M"}),
                (1, 2, {"label": "i"}),
                (2, 3, {"label": "s"}),
                (3, 4, {"label": "h"}),
                (4, 5, {"label": "a"}),
            ]
        )
        assert rpq(Regex(""), graph) == set()


class TestsForBfsRpq:
    def test_works_as_expected(self):
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "c"}),
                (0, 2, {"label": "a"}),
                (1, 2, {"label": "a"}),
                (2, 2, {"label": "b"}),
            ]
        )
        regex = Regex("a.b*")

        result = bfs_rpq(regex, graph, False, {0}, {2})
        assert result == {2}

    def test_empty_graph(self):
        graph = MultiDiGraph()
        assert bfs_rpq(Regex("Not empty regex"), graph, False) == set()
        assert bfs_rpq(Regex("Not empty regex"), graph, True) == set()

    def test_empty_query(self):
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "M"}),
                (1, 2, {"label": "i"}),
                (2, 3, {"label": "s"}),
                (3, 4, {"label": "h"}),
                (4, 5, {"label": "a"}),
            ]
        )
        assert bfs_rpq(Regex(""), graph, False) == set()
