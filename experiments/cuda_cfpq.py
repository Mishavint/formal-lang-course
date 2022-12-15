from typing import Any
import pycubool
from pyformlang.cfg import Variable, Epsilon, Terminal
from pyformlang.finite_automaton import EpsilonNFA

from project.cfg_util import create_wcnf_by_cfg

from experiments.cuda_bool_matrices import BoolMatrices

from project.ecfg import ECFG
from project.rsm import RSM


def cfpq_matrix(cfg, graph):
    cfg = create_wcnf_by_cfg(cfg)

    eps_prods: set[Variable] = set()
    term_prods: dict[Any, set[Variable]] = {}
    var_prods: set[tuple[Variable, Variable, Variable]] = set()
    for p in cfg.productions:
        match p.body:
            case [Epsilon()]:
                eps_prods.add(p.head)
            case [Terminal() as t]:
                term_prods.setdefault(t.value, set()).add(p.head)
            case [Variable() as v1, Variable() as v2]:
                var_prods.add((p.head, v1, v2))

    nodes = {n: i for i, n in enumerate(graph.nodes)}
    if len(nodes) == 0:
        return set()
    adjs: dict[Variable, pycubool.Matrix] = {
        v: pycubool.Matrix.empty((len(nodes), len(nodes))) for v in cfg.variables
    }

    for n1, n2, l in graph.edges.data("label"):
        i = nodes[n1]
        j = nodes[n2]
        for v in term_prods.setdefault(l, set()):
            adjs[v][i, j] = True

    diag = pycubool.Matrix.from_lists(
        (len(nodes), len(nodes)),
        [i for i in range(len(nodes))],
        [i for i in range(len(nodes))],
    )
    for v in eps_prods:
        adjs[v] = adjs[v].ewiseadd(diag)

    changed = True
    while changed:
        changed = False
        for h, b1, b2 in var_prods:
            nvals_old = adjs[h].nvals
            adjs[b1].mxm(adjs[b2], out=adjs[h], accumulate=True)
            changed |= adjs[h].nvals != nvals_old

    nodes = {i: n for n, i in nodes.items()}
    result = set()
    for v, adj in adjs.items():
        for i, j in adj.to_list():
            result.add((nodes[i], v, nodes[j]))
    return


def cfpq_tensor(cfg, graph):

    rsm_decomp = BoolMatrices.from_rsm(
        RSM.create_rsm_by_ecfg(ECFG.from_cfg(cfg)).minimize()
    )
    graph_decomp = BoolMatrices.from_nfa(EpsilonNFA.from_networkx(graph))
    if len(graph_decomp.states) == 0:
        return set()

    diag = pycubool.Matrix.from_lists(
        (len(graph_decomp.states), len(graph_decomp.states)),
        [i for i in range(len(graph_decomp.states))],
        [i for i in range(len(graph_decomp.states))],
    )
    for v in cfg.get_nullable_symbols():
        assert isinstance(v, Variable)
        if v.value in graph_decomp.adjs:
            graph_decomp.adjs[v.value] = graph_decomp.adjs[v.value].ewiseadd(diag)
        else:
            graph_decomp.adjs[v.value] = diag.dup()

    transitive_closure_size = 0
    while True:

        transitive_closure_indices = list(
            zip(*rsm_decomp.intersect(graph_decomp).transitive_closure_any_symbol())
        )

        if len(transitive_closure_indices) == transitive_closure_size:
            break
        transitive_closure_size = len(transitive_closure_indices)

        for i, j in transitive_closure_indices:
            r_i, r_j = i // len(graph_decomp.states), j // len(graph_decomp.states)
            s, f = rsm_decomp.states[r_i], rsm_decomp.states[r_j]
            if s.is_start and f.is_final:
                assert s.data[0] == f.data[0]
                v = s.data[0]

                g_i, g_j = i % len(graph_decomp.states), j % len(graph_decomp.states)

                graph_decomp.adjs.setdefault(
                    v,
                    pycubool.Matrix.empty(
                        (len(graph_decomp.states), len(graph_decomp.states))
                    ),
                )[g_i, g_j] = True

    result = set()
    for v, adj in graph_decomp.adjs.items():
        if isinstance(v, Variable):
            for i, j in adj.to_list():
                result.add(
                    (graph_decomp.states[i].data, v, graph_decomp.states[j].data)
                )
    return result
