from project.matrix_util import BoolMatrices
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton


class TestsForBoolMatrices:
	def test_for_transitive_closure(self):
		ndfa = NondeterministicFiniteAutomaton()
		ndfa.add_transitions(
			[
				(0, "a", 1),
				(1, "b", 2),
			]
		)
		bool_matrix = BoolMatrices(ndfa)
		transitive_closure = bool_matrix.transitive_closure()
		assert transitive_closure.sum() == transitive_closure.size