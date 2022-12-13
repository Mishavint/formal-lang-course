from functools import reduce

from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import Symbol, State
from pyformlang.finite_automaton import EpsilonNFA
from enum import Enum
from scipy import sparse
import numpy

from project.cfg_util import create_wcnf_by_cfg
from project.fa_util import get_edges_by_enfa
from project.graphs_util import create_nfa_by_graph
from project.rsm import RSM
from project.ecfg import ECFG
from project.bool_matrices import BoolMatrices, boolean_decompose_enfa


def hellings_closure(cfg: CFG, graph: MultiDiGraph):
    wcnf = create_wcnf_by_cfg(cfg)

    epsilon_productions = set()
    term_productions = {}
    var_productions = {}

    for production in wcnf.productions:
        if not production.body:
            epsilon_productions.add(production.head)
        if len(production.body) == 1:
            term = production.body[0]
            if term not in term_productions:
                term_productions[term.value] = set()
            term_productions[term.value].add(production.head)
        if len(production.body) == 2:
            var1 = Variable(production.body[0].value)
            var2 = Variable(production.body[1].value)
            if (var1, var2) not in var_productions:
                var_productions[(var1, var2)] = set()
            var_productions[(var1, var2)].add(production.head)

    res = set()

    for edge in graph.edges(data=True):
        label = edge[2]["label"]

        if label in term_productions:
            for var in term_productions[label]:
                res.add((edge[0], var, edge[1]))

    for node in graph.nodes:
        for var in epsilon_productions:
            res.add((node, var, node))

    queue = res.copy()
    while len(queue) > 0:
        temp = set()
        v1, var, v2 = queue.pop()

        for triple in res:
            if triple[2] == v1:
                if (triple[1], var) not in var_productions:
                    continue
                for vvar in var_productions[(triple[1], var)]:
                    if (triple[0], vvar, v2) not in res:
                        queue.add((triple[0], vvar, v2))
                        temp.add((triple[0], vvar, v2))
        for triple in res:
            if triple[0] == v2:
                if (var, triple[1]) not in var_productions:
                    continue
                for vvar in var_productions[(var, triple[1])]:
                    if (v1, vvar, triple[2]) not in res:
                        queue.add((v1, vvar, triple[2]))
                        temp.add((v1, vvar, triple[2]))

        res = res.union(temp)

    return res


def matrix_closure(cfg: CFG, graph: MultiDiGraph):
    wcnf = create_wcnf_by_cfg(cfg)

    epsilon_productions = set()
    term_productions = {}
    var_productions = {}

    for production in wcnf.productions:
        if not production.body:
            epsilon_productions.add(production.head)
        if len(production.body) == 1:
            term = production.body[0]
            if term not in term_productions:
                term_productions[term.value] = set()
            term_productions[term.value].add(production.head)
        if len(production.body) == 2:
            var1 = Variable(production.body[0].value)
            var2 = Variable(production.body[1].value)
            if (var1, var2) not in var_productions:
                var_productions[(var1, var2)] = set()
            var_productions[(var1, var2)].add(production.head)

    res = set()
    bd_res_dok_matrix = {}
    nodes = list(graph.nodes)

    for variable in wcnf.variables:
        bd_res_dok_matrix[variable] = sparse.dok_matrix(
            (len(nodes), len(nodes)), dtype=numpy.int8
        )

    for edge in graph.edges(data=True):
        label = edge[2]["label"]
        v1 = nodes.index(edge[0])
        v2 = nodes.index(edge[1])
        if label in term_productions:
            for var in term_productions[label]:
                bd_res_dok_matrix[var][v1, v2] = 1

    for node in graph.nodes:
        v1 = nodes.index(node)
        for var in epsilon_productions:
            bd_res_dok_matrix[var][v1, v1] = 1

    bd_res = {}
    for key, value in bd_res_dok_matrix.items():
        bd_res[key] = value.tocsr()

    while True:
        old_nnz = sum([v.getnnz() for v in bd_res.values()])

        for production in var_productions.items():
            for variable in production[1]:
                bd_res[variable] += bd_res[production[0][0]] @ bd_res[production[0][1]]

        if old_nnz == sum([v.getnnz() for v in bd_res.values()]):
            break

    for variable, matrix in bd_res.items():
        rows, cols = matrix.nonzero()
        for i in range(len(rows)):
            res.add((nodes[rows[i]], variable, nodes[cols[i]]))

    return res


def tensor_closure(cfg: CFG, graph) -> set[tuple[any, Variable, any]]:
    if cfg.productions == set():
        return set()
    automata = RSM.create_rsm_by_ecfg(ECFG.from_cfg(cfg))
    variables = cfg.variables
    terminals = cfg.terminals
    start_states = reduce(
        lambda acc, l: acc + l,
        [
            list(zip(a.start_states, [var for _ in range(len(a.start_states))]))
            for var, a in automata.var_to_automata.items()
        ],
    )
    final_states = reduce(
        lambda acc, l: acc + l,
        [
            list(zip(a.final_states, [var for _ in range(len(a.final_states))]))
            for var, a in automata.var_to_automata.items()
        ],
    )
    all_states = reduce(
        lambda acc, l: acc + l,
        [
            list(zip(a.states, [var for _ in range(len(a.states))]))
            for var, a in automata.var_to_automata.items()
        ],
    )

    # boolean decompose graph and cfg`s recursive automata
    automata_size = len(all_states)
    symbol_to_matrix = dict()
    for var in variables:
        mat = sparse.dok_matrix((automata_size, automata_size), dtype=numpy.int32)
        for enfa_var, enfa in automata.var_to_automata.items():
            for (v, symbol, u) in get_edges_by_enfa(enfa):
                if var.value == symbol.value:
                    mat[
                        all_states.index((v, enfa_var)), all_states.index((u, enfa_var))
                    ] = 1
        symbol_to_matrix[Symbol(var.value)] = mat
    for term in terminals:
        mat = sparse.dok_matrix((automata_size, automata_size), dtype=numpy.int32)
        for enfa_var, enfa in automata.var_to_automata.items():
            for (v, symbol, u) in get_edges_by_enfa(enfa):
                if term.value == symbol.value:
                    mat[
                        all_states.index((v, enfa_var)), all_states.index((u, enfa_var))
                    ] = 1
        symbol_to_matrix[Symbol(term.value)] = mat

    boolean_decomposition_automata = BoolMatrices(
        symbol_to_matrix, list(map(lambda s: State(s), all_states))
    )
    boolean_decomposition_graph = boolean_decompose_enfa(create_nfa_by_graph(graph))

    # add new edges in initial graph with epsilon variables
    result = set()

    graph_size = len(graph.nodes)
    graph_nodes = list(graph.nodes)
    for production in cfg.productions:
        if len(production.body) == 0:
            symbol = Symbol(production.head.value)
            symbols_to_matrix = boolean_decomposition_graph.symbols_to_matrix
            if symbol not in symbols_to_matrix:
                symbols_to_matrix[symbol] = sparse.dok_matrix(
                    (graph_size, graph_size), dtype=numpy.int32
                )
            for i in range(graph_size):
                symbols_to_matrix[symbol][i, i] = 1
                result.add((graph_nodes[i], production.head, graph_nodes[i]))

    # algorithm
    while True:
        result_size = len(result)
        kron_decomposition = boolean_decomposition_automata.kron(
            boolean_decomposition_graph
        )
        transitive_closure = kron_decomposition.transitive_closure()
        for (i, j) in zip(*transitive_closure.nonzero()):
            (s, x) = kron_decomposition.states()[i].value
            (f, y) = kron_decomposition.states()[j].value
            if s.value in start_states and f.value in final_states:
                symbol = Symbol(s.value[1])
                if symbol not in boolean_decomposition_graph.symbols_to_matrix:
                    boolean_decomposition_graph.symbols_to_matrix[
                        symbol
                    ] = sparse.dok_matrix((graph_size, graph_size), dtype=numpy.int32)
                boolean_decomposition_graph.symbols_to_matrix[symbol][
                    boolean_decomposition_graph.state_index(x.value),
                    boolean_decomposition_graph.state_index(y.value),
                ] = 1
                result.add((x.value, s.value[1], y.value))

        if len(result) == result_size:
            break

    return result


class CfpqAlgorithms(Enum):
    HELLINGS = hellings_closure
    MATRIX = matrix_closure
    TENSOR = tensor_closure


def cfpq(
    cfg: CFG,
    graph: MultiDiGraph,
    start_nodes: set = None,
    final_nodes: set = None,
    start_symbol: Variable = Variable("S"),
    algorithm: CfpqAlgorithms = CfpqAlgorithms.HELLINGS,
):
    if not start_nodes:
        start_nodes = set(graph.nodes)
    if not final_nodes:
        final_nodes = set(graph.nodes)

    algorithm_res = algorithm(cfg, graph)

    res = set()
    for v1, var, v2 in algorithm_res:
        if var == start_symbol and v1 in start_nodes and v2 in final_nodes:
            res.add((v1, v2))

    return res
