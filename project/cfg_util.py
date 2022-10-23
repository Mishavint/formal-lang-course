from pyformlang.cfg import CFG


def cfg_from_file(path: str):
    file = open(path)
    return CFG.from_text(file.read())


def create_wcnf_by_cfg(cfg: CFG):
    new_cfg = cfg.eliminate_unit_productions().remove_useless_symbols()
    new_prods = new_cfg._get_productions_with_only_single_terminals()
    new_prods = new_cfg._decompose_productions(new_prods)
    return CFG(start_symbol=new_cfg.start_symbol, productions=set(new_prods))
