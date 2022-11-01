import pytest

from project.cfg_util import *
from pyformlang.cfg import Terminal, Variable, Production

path = "tests/resources/cfg_files"


class TestsForReadFromFile:
    def test_works_as_expected(self):
        cfg = cfg_from_file(f"{path}/1.txt")
        assert not cfg.is_empty()
        assert cfg.terminals == {Terminal("x"), Terminal("y"), Terminal("f")}
        assert cfg.variables == {
            Variable("X"),
            Variable("Y"),
            Variable("S"),
            Variable("F"),
        }
        assert cfg.start_symbol == Variable("S")

    def test_missed_file(self):
        with pytest.raises(FileNotFoundError):
            cfg_from_file("not existing file")


class TestsForCreateWcnf:
    def test_remove_unused(self):
        cfg = cfg_from_file(f"{path}/1.txt")
        wcnf = create_wcnf_by_cfg(cfg)

        assert Variable("F") in cfg.variables
        assert Variable("F") not in wcnf.variables

    def test_decompose(self):
        cfg = cfg_from_file(f"{path}/2.txt")
        wcnf = create_wcnf_by_cfg(cfg)

        for prod in wcnf.productions:
            assert len(prod.body) <= 2
