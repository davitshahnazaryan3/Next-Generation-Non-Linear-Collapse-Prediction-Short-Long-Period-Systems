def EC8_2004(mu, T, Tc):
	"""
    Details:
    This implements the R-mu-T relationship provided in Annex B of Eurocode 8 Part 1
    Information:
    Author: Gerard J. O'Reilly
    First Version: April 2020
    Notes:
    References:
    CEN. [2004] Eurocode 8: Design of Structures for Earthquake Resistance -
    Part 1: General Rules, Seismic Actions and Rules for Buildings
    (EN 1998-1:2004), Brussels, Belgium.
    Inputs:
    mu:
    T: period
    Tc: corner period
    Returns:
    R: strength ratio
    """

	# Compute the parameters
	if T < Tc:
		R = (mu - 1) * (T / Tc) + 1
	else:
		R = mu

	# Return the outputs
	return R
