import pytest
import cfpq_data

from project.graphs_util import *


def setup_module(module):
    print("basic setup module")


def teardown_module(module):
    print("basic teardown module")


def test_1():
    assert 1 + 1 == 2


def test_2():
    assert "1" + "1" == "11"


def test3():
    get_info_from_graph("wine")
