from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from scipy import sparse


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
