def Guerrini_et_al_2017(R, *data):
	"""
	Details:
	This implements the R-mu-T relationship proposed by Guerrini et al 2017
	Information:
	Author: Gerard J. O'Reilly
	First Version: September 2022
	Notes:
	References:
	<div class="csl-entry">Guerrini, G., Graziotti, F., Penna, A., &#38;
	Magenes, G. (2017). Improved evaluation of inelastic displacement
	demands for short-period masonry structures. <i>Earthquake Engineering
	&#38; Structural Dynamics</i>, <i>46</i>(9), 1411–1430.
	https://doi.org/10.1002/eqe.2862</div>
	Inputs:
	R:
	T: period
	case: see Table II in article, (options: FD, IN, SD)
	Tc: corner period
	Returns:
	mu: ductility demand
	"""
	mu, T, case, Tc = data
	if case == "FD":
		ahyst = 0.7
		Thyst = 0.055
	elif case == "IN":
		ahyst = 0.2
		Thyst = 0.030
	elif case == "SD":
		ahyst = 0.0
		Thyst = 0.022
	else:
		raise ValueError

	b = 2.3
	c = 2.1

	return abs(mu - (R + pow(R - 1, c) / ((T / Thyst + ahyst) * pow(T / Tc, b))))

