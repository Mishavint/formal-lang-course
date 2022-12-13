import pyformlang.cfg
from project.ecfg import ECFG


class RSM:
    def __init__(self, start_symbol: pyformlang.cfg.Variable, boxes: dict):
        self.start_symbol = start_symbol
        self.boxes = boxes

    def minimize(self):
        boxes = {}
        for key, fa in self.boxes.items():
            boxes[key] = fa.minimize()
        return RSM(start_symbol=self.start_symbol, boxes=boxes)

    @property
    def var_to_automata(self):
        return self.boxes

    @classmethod
    def create_rsm_by_ecfg(self, ecfg: ECFG):
        boxes = {}
        for var, regex in ecfg.productions.items():
            boxes[var] = regex.to_epsilon_nfa().minimize()
        return RSM(start_symbol=ecfg.start_symbol, boxes=boxes)
