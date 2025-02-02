from project.cfg_util import cfg_from_file
from project.cyk import cyk
from pyformlang.cfg import CFG

path = "tests/resources/cfg_files"


class TestsForCYK:
    def test_empty_str(self):
        cfg = cfg_from_file(f"{path}/1.txt")
        res = cyk(cfg, "")

        assert not res

    def test_works_as_expected(self):
        cfg = cfg_from_file(f"{path}/1.txt")

        res = cyk(cfg, "xy")
        assert res

        res = cyk(cfg, "yx")
        assert not res

        cfg = cfg_from_file(f"{path}/4.txt")

        res = cyk(cfg, "()")
        assert res

        res = cyk(cfg, "(())")
        assert res

        res = cyk(cfg, ")(")
        assert not res

        res = cyk(cfg, "())")
        assert not res

        res = cyk(cfg, "(()")
        assert not res

        cfg = cfg_from_file(f"{path}/5.txt")

        res = cyk(cfg, "baaba")
        assert res
