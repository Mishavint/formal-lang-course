from pyformlang.cfg import CFG, Terminal, Variable
from pyformlang.regular_expression import Regex


class ECFG:
    def __init__(self, productions: dict, start_symbol):
        self.start_symbol = start_symbol
        self.productions = productions

    @classmethod
    def from_cfg(self, cfg: CFG):
        productions = {}
        start_symbol = cfg.start_symbol

        for production in cfg.productions:
            regex = Regex(
                ".".join(variable.value for variable in production.body)
                if len(production.body) > 0
                else ""
            )
            if production.head not in productions:
                productions[production.head] = regex
            else:
                productions[production.head] = productions[production.head].union(regex)

        return ECFG(productions, start_symbol)
