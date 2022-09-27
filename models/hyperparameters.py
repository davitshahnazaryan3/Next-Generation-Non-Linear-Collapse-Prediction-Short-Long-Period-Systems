def get_max_depth_min_child_weight():
	gridsearch_params = [
		(max_depth, min_child_weight)
		for max_depth in range(2, 10)
		for min_child_weight in range(6, 13)
	]
	return gridsearch_params


def get_eta():
	grid = [.3, .2, .1, .05, .01, .005]
	return grid


def get_regularization_parameters():
	grid = [
		(reg_alpha, reg_lambda)
		for reg_alpha in [0, 0.5, 1, 5]
		for reg_lambda in [0, 0.5, 1, 5]
	]
	return grid
