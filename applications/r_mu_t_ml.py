from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from plot_utils import *
from utils import export_figure, export_results


def get_medians_v2(bin_size, xs, ys):
	medians = []
	duct_range = []

	current = min(xs)
	while current < max(xs):

		data = ys[(xs >= current) & (xs < current + bin_size)]
		medians.append(np.median(data))
		duct_range.append(current + bin_size / 2)

		current += bin_size

	return medians, duct_range


class MLPredictions:
	def __init__(self, model_path, ductility, period, ah, ac, ml="xgb", dv="R"):
		self.ductility = ductility
		self.period = period
		self.ah = ah
		self.ac = ac
		self.ml = ml
		self.dv = dv

		data = pickle.load(open(model_path / f"updated_{self.dv}_{self.ml}.pickle", 'rb'))

		self.X_test = data["X_test"]
		self.y_test = data["y_test"]
		self.predictions = data["predictions"]

	def derive_sdof_shape(self):

		ultimate = 1 + (self.ductility - 1) * self.ah
		ductility_end = self.ductility + ultimate / abs(self.ac)

		sdof_x = [0, 1, self.ductility, ductility_end]
		sdof_y = [0, 1, 1 + (self.ductility - 1) * self.ah, 0]

		return sdof_x, sdof_y

	def derive_ml_data(self):

		idxs = []
		rows_x = []
		rows_y = []
		predictions_sdof = []

		for i in range(self.X_test.shape[0]):
			if round(self.X_test[i][0], 2) == self.period and \
					round(self.X_test[i][1], 2) == self.ah \
					and self.X_test[i][2] == self.ductility:

				idxs.append(i)
				rows_x.append(self.X_test[i])
				rows_y.append(self.y_test[i][0])
				predictions_sdof.append(self.predictions[i])

		ductility_dyn = np.array(rows_x)[:, -1]
		y = np.array(rows_y)
		predictions = np.array(predictions_sdof)

		return ductility_dyn, y, predictions

	def remove_all_before_softening(self, xs, ys, predictions):
		idx_to_keep = np.where(xs <= self.ductility)
		xs = xs[idx_to_keep]
		ys = ys[idx_to_keep]
		predictions = predictions[idx_to_keep]
		return xs, ys, predictions

	def calculate_medians(self, xs, ys, predictions):
		# Sort the data by ductility
		idx = np.argsort(xs)
		xs = xs[idx]
		ys = ys[idx]
		predictions = predictions[idx]

		bin_size = (max(xs) - min(xs)) / 10

		medians, duct_range = get_medians_v2(bin_size, xs, ys)
		medians_pred, _ = get_medians_v2(bin_size, xs, predictions)

		return duct_range, medians, medians_pred

	def sample_data(self, x, y, predictions, sample_size=1000):
		sample_idx_nltha = random.sample(range(len(y)), sample_size)
		x = x[sample_idx_nltha]
		y = y[sample_idx_nltha]
		predictions = predictions[sample_idx_nltha]
		return x, y, predictions

	def visualize(self, sdof_x, sdof_y, duct_range, medians, medians_pred, xs, ys):
		# Plotting
		fig, ax = plt.subplots(figsize=(3.0, 2.5))

		plt.plot(sdof_x, sdof_y, color="k")
		plt.plot(duct_range, medians, color="b", label="Cloud, median", marker="o", markersize=6)
		plt.plot(duct_range, medians_pred, color="r", label="XGB, median", marker="x", markersize=6)
		plt.scatter(xs, ys, marker="o", s=6, color="#7DCBFF", label="Cloud")

		plt.xlim(0, 5)
		plt.ylim(0, 8)
		plt.xlabel(r"Observed ductility demand, $\mu_{dyn}$", fontsize=FONTSIZE)
		plt.ylabel(r"Static strength ratio, $R$", fontsize=FONTSIZE)
		plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
		plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
		ax.tick_params(axis='both', labelsize=FONTSIZE)
		plt.legend(frameon=False, loc='best', fontsize=FONTSIZE)

		plt.show()
		export_figure(fig, filename=f"figuresComparisons/"
		                            f"{self.ml}_{self.dv}_{self.period}_{self.ductility}", filetype="svg")


