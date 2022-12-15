from itertools import product
from typing import Any, NamedTuple

import pycubool
from pyformlang.finite_automaton import EpsilonNFA


class BoolMatrices:
    class StateInfo(NamedTuple):
        data: Any
        is_start: bool
        is_final: bool

        def __eq__(self, other):
            return isinstance(other, BoolMatrices.StateInfo) and self.data == other.data

        def __hash__(self):
            return hash(self.data)

    def __init__(
        self,
        states=None,
        adjs=None,
    ):
        self.states: list[BoolMatrices.StateInfo] = states if states is not None else []
        self.adjs: dict[Any, pycubool.Matrix] = adjs if adjs is not None else {}

    @classmethod
    def from_nfa(cls, nfa: EpsilonNFA, sort_states: bool = False) -> "BoolMatrices":
        states = list(
            set(
                cls.StateInfo(
                    data=st.value,
                    is_start=st in nfa.start_states,
                    is_final=st in nfa.final_states,
                )
                for st in nfa.states
            )
        )
        if sort_states:
            states = sorted(states, key=lambda st: st.data)

        adjs = {}
        transitions = nfa.to_dict()
        for n_from in transitions:
            for symbol, ns_to in transitions[n_from].items():
                adj = adjs.setdefault(
                    symbol.value, pycubool.Matrix.empty((len(states), len(states)))
                )
                beg_index = next(i for i, s in enumerate(states) if s.data == n_from)
                for n_to in ns_to if isinstance(ns_to, set) else {ns_to}:
                    end_index = next(i for i, s in enumerate(states) if s.data == n_to)
                    adj[beg_index, end_index] = True

        return cls(states, adjs)

    @classmethod
    def from_rsm(cls, rsm, sort_states: bool = False) -> "BoolMatrices":
        states = list(
            {
                cls.StateInfo(
                    data=(var, st.value),
                    is_start=st in nfa.start_states,
                    is_final=st in nfa.final_states,
                )
                for var, nfa in rsm.boxes.items()
                for st in nfa.states
            }
        )
        if sort_states:
            states.sort(key=lambda st: (st.data[0].value, st.data[1]))

        adjs = {}
        for var, nfa in rsm.boxes.items():
            transitions = nfa.to_dict()
            for n_from in transitions:
                for symbol, ns_to in transitions[n_from].items():
                    adj = adjs.setdefault(
                        symbol.value, pycubool.Matrix.empty((len(states), len(states)))
                    )
                    start_index = next(
                        i for i, s in enumerate(states) if s.data == (var, n_from)
                    )
                    for n_to in ns_to if isinstance(ns_to, set) else {ns_to}:
                        end_index = next(
                            i for i, s in enumerate(states) if s.data == (var, n_to)
                        )
                        adj[start_index, end_index] = True

        return cls(states, adjs)

    def intersect(self, other: "BoolMatrices") -> "BoolMatrices":
        states = [
            self.StateInfo(
                data=(st1.data, st2.data),
                is_start=st1.is_start and st2.is_start,
                is_final=st1.is_final and st2.is_final,
            )
            for st1, st2 in product(self.states, other.states)
        ]
        if len(states) == 0:
            return BoolMatrices([], {})

        adjs = {}
        for symbol in set(self.adjs.keys()).union(set(other.adjs.keys())):
            if symbol in self.adjs and symbol in other.adjs:
                adjs[symbol] = self.adjs[symbol].kronecker(other.adjs[symbol])
            else:
                adjs[symbol] = pycubool.Matrix.empty((len(states), len(states)))

        return BoolMatrices(states, adjs)

    def transitive_closure_any_symbol(self) -> tuple[list[int], list[int]]:
        if len(self.states) == 0:
            return [], []

        adj_all = pycubool.Matrix.empty((len(self.states), len(self.states)))
        for adj in self.adjs.values():
            adj_all = adj_all.ewiseadd(adj)

        while True:
            prev_path_num = adj_all.nvals
            adj_all.mxm(adj_all, out=adj_all, accumulate=True)
            if prev_path_num == adj_all.nvals:
                break

        return adj_all.to_lists()

    def _direct_sum(self, other: "BoolMatrices") -> "BoolMatrices":
        states = self.states + other.states

        adjs = {}
        for symbol in set(self.adjs.keys()).intersection(set(other.adjs.keys())):
            dsum = pycubool.Matrix.empty((len(states), len(states)))
            for i, j in self.adjs[symbol]:
                dsum[i, j] = True
            for i, j in other.adjs[symbol]:
                dsum[len(self.states) + i, len(self.states) + j] = True
            adjs[symbol] = dsum

        return BoolMatrices(states, adjs)

    def constrained_bfs(self, constraint: "BoolMatrices", separated: bool = False):
        n = len(constraint.states)

        direct_sum = constraint._direct_sum(self)

        start_states_indices = [i for i, st in enumerate(self.states) if st.is_start]
        init_front = (
            _init_bfs_front(self.states, constraint.states)
            if not separated
            else _init_separated_bfs_front(
                self.states, constraint.states, start_states_indices
            )
        )

        visited = pycubool.Matrix.empty(init_front.shape)

        while True:
            old_visited_nvals = visited.nvals

            for adj in direct_sum.adjs.values():
                front_part = (
                    visited.mxm(adj) if init_front is None else init_front.mxm(adj)
                )
                visited = visited.ewiseadd(_transform_front_part(front_part, n))

            init_front = None

            if visited.nvals == old_visited_nvals:
                break

        results = set()
        for i, j in visited.to_list():
            if j >= n and constraint.states[i % n].is_final:
                self_st_index = j - n
                if self.states[self_st_index].is_final:
                    results.add(
                        self_st_index
                        if not separated
                        else (start_states_indices[i // n], self_st_index)
                    )
        return results


def _init_bfs_front(
    self_states,
    constr_states,
    self_start_indices=None,
):
    front = pycubool.Matrix.empty(
        (len(constr_states), len(constr_states) + len(self_states))
    )

    if self_start_indices is None:
        self_start_indices = [j for j, st in enumerate(self_states) if st.is_start]

    for i, st in enumerate(constr_states):
        if st.is_start:
            front[i, i] = True
            for j in self_start_indices:
                front[i, len(constr_states) + j] = True

    return front


def _init_separated_bfs_front(self_states, constr_states, start_states_indices):
    fronts = [
        _init_bfs_front(
            self_states,
            constr_states,
            self_start_indices=[st_i],
        )
        for st_i in start_states_indices
    ]

    if len(fronts) == 0:
        return pycubool.Matrix.empty(
            (len(constr_states), len(constr_states) + len(self_states))
        )

    result = pycubool.Matrix.empty(
        (len(fronts) * len(constr_states), len(constr_states) + len(self_states))
    )
    vstack_helper = pycubool.Matrix.empty((len(fronts), 1))
    for i, front in enumerate(fronts):
        vstack_helper.build(rows={i}, cols={0})
        result = result.ewiseadd(vstack_helper.kronecker(front))
    return result


def _transform_front_part(front_part, constr_states_num):
    transformed_front_part = pycubool.Matrix.empty(front_part.shape)
    for i, j in front_part.to_list():
        if j < constr_states_num:
            non_zero_row_right = front_part[i : i + 1, constr_states_num:]
            if non_zero_row_right.nvals > 0:
                row_shift = i // constr_states_num * constr_states_num
                transformed_front_part[row_shift + j, j] = True
                for _, r_j in non_zero_row_right:
                    transformed_front_part[
                        row_shift + j, constr_states_num + r_j
                    ] = True
    return transformed_front_part
