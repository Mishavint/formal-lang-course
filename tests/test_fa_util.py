from project.fa_util import *
from project.graphs_util import *


class TestsForCreateMinimumFaFunction:
	def test_is_dfa(self):
		dfa = create_minimum_dfa(Regex(""))
		assert dfa.is_deterministic()
		assert dfa.is_empty()

	def test_for_union(self):
		dfa = create_minimum_dfa(Regex("a|c"))
		assert dfa.accepts([Symbol("a")])
		assert dfa.accepts([Symbol("c")])
		assert not dfa.accepts([Symbol("b")])
		assert not dfa.accepts([Symbol("a"), Symbol("c")])

	def test_for_concatenation(self):
		dfa = create_minimum_dfa(Regex("a c"))
		assert dfa.accepts([Symbol("a"), Symbol("c")])
		assert not dfa.accepts([Symbol("ac")])
		assert not dfa.accepts([Symbol("a c")])

	def test_for_kleene(self):
		dfa = create_minimum_dfa(Regex("a*"))
		assert dfa.accepts([Symbol("a")])
		assert dfa.accepts([Symbol("a"), Symbol("a")])

	def test_for_parentheses(self):
		dfa = create_minimum_dfa(Regex("a (b|c)"))
		assert dfa.accepts([Symbol("a"), Symbol("b")])
		assert dfa.accepts([Symbol("a"), Symbol("c")])
		assert not dfa.accepts([Symbol("a"), Symbol("bc")])
		assert not dfa.accepts([Symbol("a"), Symbol("b"), Symbol("c")])


class TestsForCreateNdfaByGraph:
	def test_is_ndfa(self):
		graph_for_test = get_graph_by_name("generations")
		assert not create_ndfa_by_graph(graph=graph_for_test).is_deterministic()

	def test_works_as_expected(self):
		two_cycles_graph = create_two_cycles_graph(1, 1, "a", "b")
		actual_ndfa = create_ndfa_by_graph(two_cycles_graph, start_states={0}, final_states={2})

		expected_ndfa = NondeterministicFiniteAutomaton()
		expected_ndfa.add_start_state(State(0))
		expected_ndfa.add_final_state(State(2))

		expected_ndfa.add_transition(State(1), Symbol("a"), State(0))
		expected_ndfa.add_transition(State(0), Symbol("a"), State(1))
		expected_ndfa.add_transition(State(2), Symbol("b"), State(0))
		expected_ndfa.add_transition(State(0), Symbol("b"), State(2))

		assert actual_ndfa.is_equivalent_to(expected_ndfa)
