from pyformlang.cfg import Variable
from pyformlang.regular_expression import Regex

from project.ecfg import ECFG
from project.cfg_util import cfg_from_file
from project.rsm import RSM


class TestsForRSM:
    def test_works_as_expected(self):
        cfg = cfg_from_file("resources/cfg_files/3.txt")
        ecfg = ECFG.from_cfg(cfg)
        rsm = RSM.create_rsm_by_ecfg(ecfg)

        assert rsm.boxes[Variable("Y")] == Regex("y").to_epsilon_nfa()
        assert rsm.boxes[Variable("X")] == Regex("x").to_epsilon_nfa()
        assert rsm.boxes[Variable("S")] == Regex("X.Y").to_epsilon_nfa()
        assert rsm.boxes[Variable("F")] == Regex("f|g").to_epsilon_nfa()
