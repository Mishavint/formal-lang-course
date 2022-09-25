from project.fa_util import create_ndfa_by_graph, create_minimum_dfa
from project.matrix_util import BoolMatrices
from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex


def rpq(
		regex: Regex,
		graph: MultiDiGraph,
		start_nodes: set = None,
		final_nodes: set = None
) -> set:
	"""
		returns set of nodes that we can reach by regex

	Parameters
	----------
	regex: Regex
		needed regex
	graph: MultiDiGraph
		needed graph
	start_nodes: set()
		set of start nodes
	final_nodes: set()
		set of final nodes
	"""
	ndfa = create_ndfa_by_graph(graph, start_nodes, final_nodes)
	dfa = create_minimum_dfa(regex)

	bool_matrix_for_graph = BoolMatrices(ndfa)
	bool_matrix_for_query = BoolMatrices(dfa)

	bool_matrix_intersected = bool_matrix_for_graph.intersect(bool_matrix_for_query)

	start_states = bool_matrix_intersected.get_start_states()
	final_states = bool_matrix_intersected.get_final_states()

	transitive = bool_matrix_intersected.transitive_closure()

	res = set()
	for first_state, second_state in zip(*transitive.nonzero()):
		if first_state in start_states and second_state in final_states:
			res.add(
				(first_state // bool_matrix_for_query.num_of_states,
				 second_state // bool_matrix_for_graph.num_of_states)
			)

	return res
