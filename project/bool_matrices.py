from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from pyformlang.regular_expression import Regex
from scipy import sparse
from project.fa_util import create_ndfa_by_graph, create_minimum_dfa


class BoolMatrices:
    """
    Class that presents NFA as Boolean Matrix
    """

    def __init__(self, nfa: NondeterministicFiniteAutomaton = None):
        if nfa is not None:
            self.states = nfa.states
            self.start_states = nfa.start_states
            self.final_states = nfa.final_states
            self.num_of_states = len(self.states)
            self.states_indices = {
                state: index for index, state in enumerate(nfa.states)
            }
            self.bool_matrices = self.init_bool_matrices_by_nfa(nfa)
        else:
            self.states = set()
            self.start_states = set()
            self.final_states = set()
            self.num_of_states = 0
            self.states_indices = dict()
            self.bool_matrices = dict()

    def init_bool_matrices_by_nfa(self, nfa: NondeterministicFiniteAutomaton):
        """
                                        Creates boolean matrices by nfa

        Parameters
        ----------
        nfa: NondeterministicFiniteAutomaton
                                        needed nfa
        """
        res = dict()
        for first_state, transition in nfa.to_dict().items():
            for symbol, second_states in transition.items():
                if not isinstance(second_states, set):
                    second_states = {second_states}
                for state in second_states:
                    first_index = self.states_indices[first_state]
                    second_index = self.states_indices[state]
                    if symbol not in res:
                        res[symbol] = sparse.csr_matrix(
                            (self.num_of_states, self.num_of_states), dtype=bool
                        )
                    res[symbol][first_index, second_index] = True
        return res

    def to_nfa(self):
        nfa = NondeterministicFiniteAutomaton()
        for symbol, bm in self.bool_matrices.items():
            for first_state, second_state in zip(*bm.nonzero()):
                nfa.add_transition(first_state, symbol, second_state)

        for state in self.start_states:
            nfa.add_start_state(state)

        for state in self.final_states:
            nfa.add_final_state(state)

        return nfa

    def intersect(self, other: "BoolMatrices"):
        res = BoolMatrices()
        res.num_of_states = self.num_of_states * other.num_of_states
        symbols = self.bool_matrices.keys() & other.bool_matrices.keys()

        for symbol in symbols:
            res.bool_matrices[symbol] = sparse.kron(
                self.bool_matrices[symbol], other.bool_matrices[symbol], format="csr"
            )

        for first_state, first_index in self.states_indices.items():
            for second_state, second_index in other.states_indices.items():
                state_index = first_index * other.num_of_states + second_index
                res.states_indices[state_index] = state_index

                state = state_index
                if (
                    first_state in self.start_states
                    and second_state in other.start_states
                ):
                    res.start_states.add(state)
                if (
                    first_state in self.final_states
                    and second_state in other.final_states
                ):
                    res.final_states.add(state)

        return res

    def transitive_closure(self):
        if len(self.bool_matrices) == 0:
            return sparse.csr_matrix((0, 0), dtype=bool)
        tc = sum(self.bool_matrices.values())

        prev = tc.nnz
        curr = 0

        while prev != curr:
            tc += tc @ tc
            prev = curr
            curr = tc.nnz

        return tc

    def get_start_states(self):
        return self.start_states

    def get_final_states(self):
        return self.final_states

    def constraint_bfs(self, other: "BoolMatrices", separate: bool = False):
        direct_sum = other.direct_sum(self)
        n = self.num_of_states
        k = other.num_of_states

        start_states_indices = [
            index
            for index, state in enumerate(self.states)
            if state in self.start_states
        ]
        final_states_indices = [
            index
            for index, state in enumerate(self.states)
            if state in self.final_states
        ]
        other_final_states_indices = [
            index
            for index, state in enumerate(other.states)
            if state in other.final_states
        ]

        if not separate:
            front = self.make_front(other)
        else:
            front = self.make_separated_front(other)

        visited = sparse.csr_matrix(front.shape)

        while True:
            old_visited = visited.copy()

            for _, matrix in direct_sum.bool_matrices.items():
                if front is not None:
                    front2 = front @ matrix
                else:
                    front2 = visited @ matrix

                visited += self.transform_front(front2, other)

            front = None

            if visited.nnz == old_visited.nnz:
                break

        result = set()
        for i, j in zip(*visited.nonzero()):
            if j >= k and i % k in other_final_states_indices:
                if j - k in final_states_indices:
                    if not separate:
                        result.add(j - k)
                    else:
                        result.add((start_states_indices[i // n], j - k))

        return result

    def direct_sum(self, other: "BoolMatrices"):
        result = BoolMatrices()
        bool_matrices = {}
        symbols = self.bool_matrices.keys() & other.bool_matrices.keys()

        for symbol in symbols:
            bool_matrices[symbol] = sparse.bmat(
                [
                    [self.bool_matrices[symbol], None],
                    [None, other.bool_matrices[symbol]],
                ]
            )

        start_states = {
            State(state.value + self.num_of_states) for state in other.start_states
        }
        final_states = {
            State(state.value + self.num_of_states) for state in other.final_states
        }

        for _, first_index in self.states_indices.items():
            for _, second_index in other.states_indices.items():
                state = first_index * other.num_of_states + second_index
                result.states_indices[state] = state

        result.states = self.states | set(
            (State(state.value + self.num_of_states) for state in other.states)
        )
        result.num_of_states = self.num_of_states + other.num_of_states
        result.start_states = self.start_states | start_states
        result.final_states = self.final_states | final_states

        result.bool_matrices = bool_matrices

        return result

    def make_front(self, other: "BoolMatrices"):
        n = self.num_of_states
        k = other.num_of_states

        front = sparse.lil_matrix((k, n + k))

        right_part = sparse.lil_array(
            [[state in self.start_states for state in self.states]]
        )

        for _, index in other.states_indices.items():
            front[index, index] = True
            front[index, k:] = right_part

        return front.tocsr()

    def make_separated_front(self, other: "BoolMatrices"):
        start_indices = {
            index
            for index, state in enumerate(self.states)
            if state in self.start_states
        }
        fronts = [self.make_front(other) for _ in start_indices]

        if len(fronts) > 0:
            return sparse.csr_matrix(sparse.vstack(fronts))
        else:
            return sparse.csr_matrix(
                (other.num_of_states, other.num_of_states + self.num_of_states)
            )

    def transform_front(self, part: sparse.csr_matrix, other: "BoolMatrices"):
        transformed_part = sparse.lil_array(part.shape)

        for i, j in zip(*part.nonzero()):
            if j < other.num_of_states:
                non_zero_right = part.getrow(i).tolil()[[0], other.num_of_states :]

                if non_zero_right.nnz > 0:
                    shift_row = i // other.num_of_states * other.num_of_states
                    transformed_part[shift_row + j, j] = 1
                    transformed_part[
                        [shift_row + j], other.num_of_states :
                    ] += non_zero_right

        return transformed_part.tocsr()
