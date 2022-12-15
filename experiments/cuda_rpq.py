import enum

from project.fa_util import create_ndfa_by_graph, create_minimum_dfa
from experiments.cuda_bool_matrices import BoolMatrices


class BfsMode(enum.Enum):
    FIND_COMMON_REACHABLE_SET = enum.auto()
    FIND_REACHABLE_FOR_EACH_START = enum.auto()


def rpq_by_tensor(
    query,
    graph,
    starts=None,
    finals=None,
):
    graph_decomp = BoolMatrices.from_nfa(create_ndfa_by_graph(graph, starts, finals))
    query_decomp = BoolMatrices.from_nfa(create_minimum_dfa(query))

    intersection = graph_decomp.intersect(query_decomp)
    transitive_closure_indices = intersection.transitive_closure_any_symbol()

    results = set()
    for n_from_i, n_to_i in zip(*transitive_closure_indices):
        n_from = intersection.states[n_from_i]
        n_to = intersection.states[n_to_i]
        if n_from.is_start and n_to.is_final:
            beg_graph_node = n_from.data[0]
            end_graph_node = n_to.data[0]
            results.add((beg_graph_node, end_graph_node))
    return results


def rpq_by_bfs(
    query,
    graph,
    starts=None,
    finals=None,
    mode: BfsMode = BfsMode.FIND_COMMON_REACHABLE_SET,
):
    graph_decomp = BoolMatrices.from_nfa(create_ndfa_by_graph(graph, starts, finals))
    query_decomp = BoolMatrices.from_nfa(create_minimum_dfa(query))

    result_indices = graph_decomp.constrained_bfs(
        query_decomp, separated=mode == BfsMode.FIND_REACHABLE_FOR_EACH_START
    )

    match mode:
        case BfsMode.FIND_COMMON_REACHABLE_SET:
            return {graph_decomp.states[i].data for i in result_indices}
        case BfsMode.FIND_REACHABLE_FOR_EACH_START:
            return {
                (graph_decomp.states[i].data, graph_decomp.states[j].data)
                for i, j in result_indices
            }
