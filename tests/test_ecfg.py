from pyformlang.cfg import Variable
from pyformlang.finite_automaton import Symbol

from project.ecfg import ECFG
from project.cfg_util import cfg_from_file


class TestsForECFG:
    def test_works_as_expected(self):
        cfg = cfg_from_file("tests/resources/cfg_files/3.txt")
        ecfg = ECFG.from_cfg(cfg)

        sons = ecfg.productions[Variable("F")].sons

        assert sons[0].head.value == Symbol("f") or Symbol("g")
        assert sons[1].head.value == Symbol("f") or Symbol("g")

    def test_sames(self):
        cfg = cfg_from_file("resources/cfg_files/3.txt")
        ecfg = ECFG.from_cfg(cfg)

        variables = []
        for production in ecfg.productions.items():
            assert production[0] not in variables
            variables.append(production[0])
