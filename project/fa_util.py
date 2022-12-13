from typing import Set
from pyformlang.regular_expression import *
from pyformlang.finite_automaton import *
from networkx import MultiGraph
from scipy import sparse


def create_minimum_dfa(regex: Regex) -> DeterministicFiniteAutomaton:
    """
    Creates minimum deterministic finite automaton
    Parameters
    ----------
    regex : Regex
                    needed regex

    Returns
    -------
    DeterministicFiniteAutomaton
                    created dfa
    """
    enfa = regex.to_epsilon_nfa()

    return enfa.to_deterministic().minimize()


def create_ndfa_by_graph(
    graph: MultiGraph, start_states: Set[int] = None, final_states: Set[int] = None
) -> NondeterministicFiniteAutomaton:
    """
                    Creates a NonDeterministic Finite Automation by graph

    Parameters
    ----------
    graph: MultiGraph
                    needed graph
    start_states: Set[int]
                    Set of start edges
    final_states: Set[int]
                    Set of final edges

    Returns
    -------
    NondeterministicFiniteAutomaton
                    NDFA
    """
    ndfa = NondeterministicFiniteAutomaton()
    for edge in graph.edges(data=True):
        ndfa.add_transition(edge[0], edge[2]["label"], edge[1])

    if (start_states and final_states) is None:
        for state in ndfa.states:
            ndfa.add_start_state(state)
            ndfa.add_final_state(state)

    if start_states:
        for state in start_states:
            ndfa.add_start_state(State(state))

    if final_states:
        for state in final_states:
            ndfa.add_final_state(State(state))

    return ndfa


def get_edges_by_enfa(enfa: EpsilonNFA):
    res = set()
    for (u, symbol_and_vs) in enfa.to_dict().items():
        for (symbol, vs) in symbol_and_vs.items():
            if not type(vs) is set:
                res.add((u, symbol, vs))
            else:
                for v in vs:
                    res.add((u, symbol, v))

    return res
