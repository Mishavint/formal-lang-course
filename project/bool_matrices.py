import numpy
from pyformlang.finite_automaton import State, EpsilonNFA
from scipy import sparse

__all__ = ["BoolMatrices", "boolean_decompose_enfa"]


class BoolMatrices:
    """
    Class that presents NFA as Boolean Matrix
    """

    def __init__(self, symbols_to_matrix, states):
        self.symbols_to_matrix = symbols_to_matrix
        self.states_list = states
        states_num = len(states)
        for symbol in symbols_to_matrix.keys():
            symbols_to_matrix[symbol] = symbols_to_matrix[symbol].todok()
        for matrix in symbols_to_matrix.values():
            assert (states_num, states_num) == matrix.get_shape()

    def __eq__(self, other):
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        if not set(self.states()) == set(other.states()):
            return False
        if not set(self_dict.keys()) == set(other_dict.keys()):
            return False
        for i in self_dict.keys():
            nonzero_self = set(zip(*self_dict[i].nonzero()))
            nonzero2_other = set(zip(*other_dict[i].nonzero()))
            if not nonzero_self == nonzero2_other:
                return False
        return True

    def states_count(self) -> int:
        return len(self.states_list)

    def state_index(self, state) -> int:
        return self.states_list.index(state)

    def states(self) -> list[State]:
        return self.states_list

    _convert_to_spmatrix = lambda mat: mat.tocsr()

    def to_dict(self):
        d = dict()
        for (symbol, matrix) in self.symbols_to_matrix.items():
            d[symbol] = BoolMatrices._convert_to_spmatrix(matrix)
        return d

    def kron(self, other: "BoolMatrices") -> "BoolMatrices":
        intersection_decomposition = dict()
        dict1 = self.to_dict()
        dict2 = other.to_dict()
        symbols = set(dict1.keys()).union(set(dict2.keys()))
        for symbol in symbols:
            if symbol in dict1:
                coo_matrix1 = dict1[symbol]
            else:
                coo_matrix1 = sparse.coo_matrix(
                    (self.states_count(), self.states_count())
                )

            if symbol in dict2:
                coo_matrix2 = dict2[symbol]
            else:
                coo_matrix2 = sparse.coo_matrix(
                    (other.states_count(), other.states_count())
                )

            intersection_decomposition[symbol] = sparse.kron(
                BoolMatrices._convert_to_spmatrix(coo_matrix1),
                BoolMatrices._convert_to_spmatrix(coo_matrix2),
            )

        intersection_states = list()
        for state1 in self.states():
            for state2 in other.states():
                intersection_states.append(State((state1, state2)))

        return BoolMatrices(intersection_decomposition, intersection_states)

    def transitive_closure(self) -> sparse.spmatrix:
        """
        :return: adjacency matrix of states corresponding to transitive closure
        """
        adjacency_matrix = sum(
            self.symbols_to_matrix.values(),
            sparse.coo_matrix((self.states_count(), self.states_count())),
        )

        adjacency_matrix = BoolMatrices._convert_to_spmatrix(adjacency_matrix)

        last_values_count = 0
        while last_values_count != adjacency_matrix.nnz:
            last_values_count = adjacency_matrix.nnz
            adjacency_matrix += adjacency_matrix @ adjacency_matrix

        return adjacency_matrix

    def direct_sum(self, other: "BoolMatrices") -> "BoolMatrices":
        direct_sum_decomposition = dict()
        dict1 = self.to_dict()
        dict2 = other.to_dict()
        symbols = set(dict1.keys()).union(set(dict2.keys()))
        self_states_count = self.states_count()
        other_states_count = other.states_count()
        for symbol in symbols:
            if symbol in dict1:
                coo_matrix1 = dict1[symbol].tocsr()
            else:
                coo_matrix1 = sparse.csr_matrix((self_states_count, self_states_count))

            if symbol in dict2:
                coo_matrix2 = dict2[symbol].tocsr()
            else:
                coo_matrix2 = sparse.csr_matrix(
                    (other_states_count, other_states_count)
                )
            direct_sum_decomposition[symbol] = sparse.coo_matrix(
                (
                    self_states_count + other_states_count,
                    self_states_count + other_states_count,
                )
            )
            data = [coo_matrix1[i, j] for (i, j) in zip(*coo_matrix1.nonzero())] + [
                coo_matrix2[i, j] for (i, j) in zip(*coo_matrix2.nonzero())
            ]
            row = [i for (i, _) in zip(*coo_matrix1.nonzero())] + [
                self_states_count + i for (i, _) in zip(*coo_matrix2.nonzero())
            ]
            col = [j for (_, j) in zip(*coo_matrix1.nonzero())] + [
                self_states_count + j for (_, j) in zip(*coo_matrix2.nonzero())
            ]
            shape_width = self_states_count + other_states_count
            direct_sum_decomposition[symbol] = sparse.coo_matrix(
                (data, (row, col)), shape=(shape_width, shape_width)
            )

        return BoolMatrices(direct_sum_decomposition, self.states() + other.states())


def boolean_decompose_enfa(enfa: EpsilonNFA) -> "BoolMatrices":
    states_data = list(enfa.states)
    boolean_decompose = dict()
    for (u, symbol_and_vs) in enfa.to_dict().items():
        for (symbol, vs) in symbol_and_vs.items():
            if symbol not in boolean_decompose:
                boolean_decompose[symbol] = list()
            if not type(vs) is set:  # vs is one state in this case
                boolean_decompose[symbol].append(
                    (states_data.index(u), states_data.index(vs))
                )
            else:
                for v in vs:
                    boolean_decompose[symbol].append(
                        (states_data.index(u), states_data.index(v))
                    )

    states_num = len(enfa.states)
    coo_matrices = dict()
    for (symbol, edges) in boolean_decompose.items():
        row = numpy.array([i for (i, _) in edges])
        col = numpy.array([j for (_, j) in edges])
        data = numpy.array([1 for _ in range(len(edges))])
        coo_matrices[symbol] = sparse.coo_matrix(
            (data, (row, col)), shape=(states_num, states_num)
        )

    return BoolMatrices(coo_matrices, states_data)
