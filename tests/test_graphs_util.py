import os

import pytest
from project.graphs_util import *


class TestsForGetFunction:
    def test_wrong_graph_name_for_get_function(self):
        with pytest.raises(FileNotFoundError):
            get_graph_by_name("lololo")

    def test_works_as_expected_for_get_function(self):
        graph = get_graph_by_name("generations")
        assert graph.number_of_nodes() == 129
        assert graph.number_of_edges() == 273


class TestsForGetInfo:
    def test_wrong_graph_name_for_get_info_function(self):
        with pytest.raises(FileNotFoundError):
            get_info_from_graph("lololo")

    def test_works_as_expected_for_get_info_function(self):
        result = get_info_from_graph("generations")
        assert result[0] == 129
        assert result[1] == 273


class TestsForCreateTwoCycles:
    def test_works_as_expected_for_create_two_cycles_function(self):
        result = create_two_cycles_graph(1, 1, "11", "22")
        assert result.number_of_nodes() == 3
        assert result.number_of_edges() == 4


class TestsForSaveToDotFile:
    def test_file_with_wrong_exception_for_save_graph(self):
        with pytest.raises(Exception):
            save_graph_to_dot_file(get_graph_by_name("generations"), "lololo.not_dot")

    def test_works_as_expected(self):
        save_graph_to_dot_file(
            create_two_cycles_graph(1, 1, "11", "11"), "file_for_test.dot"
        )
        assert (
            open("file_for_test.dot", "r").read()
            == """digraph  {
1;
0;
2;
1 -> 0  [key=0, label=11];
0 -> 1  [key=0, label=11];
0 -> 2  [key=0, label=11];
2 -> 0  [key=0, label=11];
}
"""
        )
        os.remove("file_for_test.dot")


class TestsForCreateAndSave:
    def test_file_with_wrong_exception_for_save_graph(self):
        with pytest.raises(Exception):
            create_and_save_graph(1, 1, "11", "22", "lololo.not_dot")

    def test_workd_as_expected(self):
        create_and_save_graph(1, 1, "11", "11", "file_for_test.dot")
        assert (
            open("file_for_test.dot", "r").read()
            == """digraph  {
1;
0;
2;
1 -> 0  [key=0, label=11];
0 -> 1  [key=0, label=11];
0 -> 2  [key=0, label=11];
2 -> 0  [key=0, label=11];
}
"""
        )
        os.remove("file_for_test.dot")
