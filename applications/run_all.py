from pathlib import Path
import pickle
import xgboost as xgb
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from plot_utils import *
from utils import export_figure
from applications.r_mu_t_ml import MLPredictions
from AnalyticalModels.ec8 import EC8_2004
from AnalyticalModels.guerrini import Guerrini_et_al_2017
from AnalyticalModels.krawinkler_nassar import Krawinkler_Nassar_1992
from AnalyticalModels.miranda import Miranda_1993
from AnalyticalModels.newmark_hall import Newmark_Hall_1992
from AnalyticalModels.vidic import Vidic_et_al_1994


class RMUTModels:
	soil = {
		"A": [0.4, 800, np.inf],
		"B": [0.5, 360, 800],
		"C": [0.6, 180, 360],
	}

	def __init__(self, x_range, period, ductility, ah=0.02):
		self.x_range = x_range
		self.period = period
		self.ductility = ductility
		self.ah = ah
		self.ac = -0.5
		self.soil_class = "B"

	def visualize_single(self, sdof_x, sdof_y, duct_range, medians, medians_pred, xs, ys):
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

	def get_ml_outputs(self, model_path, sample=0, pflag=False):
		ml = MLPredictions(model_path, self.ductility, self.period, self.ah,
		                   self.ac)
		sdof_x, sdof_y = ml.derive_sdof_shape()
		ductility_dyn, y, predictions = ml.derive_ml_data()
		ductility_dyn, y, predictions = ml.remove_all_before_softening(ductility_dyn, y, predictions)
		duct_range, medians, medians_pred = ml.calculate_medians(ductility_dyn, y, predictions)
		if sample > 0:
			ductility_dyn, y, predictions = ml.sample_data(ductility_dyn, y, predictions, sample_size=sample)

		if pflag:
			self.visualize_single(sdof_x, sdof_y, duct_range, medians, medians_pred, ductility_dyn, y)

		return sdof_x, sdof_y, duct_range, medians, medians_pred, ductility_dyn, y

	def xgb_single_prediction(self, model_path):
		model = joblib.load(model_path / "R_xgb_v3.sav")
		data = pickle.load(open(model_path / f"R_xgb_v3.pickle", 'rb'))
		scaler = data["scaler"]

		df = {
			"period": [self.period],
			"hardening_ratio": [self.ah],
			"ductility": [self.ductility],
			"actual_ductility_end": [self.ductility]
		}

		df = pd.DataFrame.from_dict(df)
		x = scaler.transform(df)

		dx = xgb.DMatrix(x)
		prediction = np.expm1(model.predict(dx))
		return prediction

	def xgb_make_predictions(self, model_path):
		model = joblib.load(model_path / "R_xgb_v3.sav")
		data = pickle.load(open(model_path / f"R_xgb_v3.pickle", 'rb'))
		scaler = data["scaler"]

		predictions = []
		for dyn in self.x_range:
			# X = [period, hardening_ratio, ductility, ductility_dyn]
			df = {
				"period": [self.period],
				"hardening_ratio": [self.ah],
				"ductility": [self.ductility],
				"actual_ductility_end": [dyn]
			}
			df = pd.DataFrame.from_dict(df)
			x = scaler.transform(df)

			dx = xgb.DMatrix(x)
			prediction = np.expm1(model.predict(dx))
			predictions.append(prediction)

		return predictions

	def ec8_predictions(self):
		Tc = self.soil[self.soil_class][0]
		predictions = []
		for dyn in self.x_range:
			predictions.append(EC8_2004(dyn, self.period, Tc))
		return predictions

	def krawinkler_predictions(self):
		predictions = []
		for dyn in self.x_range:
			predictions.append(Krawinkler_Nassar_1992(dyn, self.period, self.ah*100))
		return predictions

	def miranda_predictions(self):
		# Assume
		Tg = 1.0
		site = "rock"

		predictions = []
		for dyn in self.x_range:
			predictions.append(Miranda_1993(dyn, self.period, site, Tg))
		return predictions

	def newmark_predictions(self):
		Tc = 0.5
		Tcc = .4

		predictions = []
		for dyn in self.x_range:
			predictions.append(Newmark_Hall_1992(dyn, self.period, Tcc, Tc))
		return predictions

	def vidic_predictions(self):
		Tc = self.soil[self.soil_class][0]

		predictions = []
		for dyn in self.x_range:
			predictions.append(Vidic_et_al_1994(dyn, self.period, Tc))
		return predictions

	def guerrini_predictions(self):
		Tc = self.soil[self.soil_class][0]
		case = "FD"

		predictions = []
		for dyn in self.x_range:
			data = (dyn, self.period, case, Tc)

			y = fsolve(Guerrini_et_al_2017, x0=np.array([10]), factor=0.1, args=data)

			predictions.append(y)

		return predictions

	def plot_all_cloud(self, model_path, legend=False):
		# XGB
		sdof_x, sdof_y, duct_range, medians, medians_pred, ductility_dyn, y = self.get_ml_outputs(model_path)

		xgb_predictions = r_mu_t.xgb_make_predictions(model_path)
		ec8 = self.ec8_predictions()
		krawinkler = self.krawinkler_predictions()
		miranda = self.miranda_predictions()
		newmark = self.newmark_predictions()
		vidic = self.vidic_predictions()
		guerrini = self.guerrini_predictions()

		# Plotting
		fig, ax = plt.subplots(figsize=(3.0, 2.5))

		# SDOF
		plt.plot(sdof_x, sdof_y, color="k", label="SDOF")
		# Cloud data
		plt.scatter(ductility_dyn, y, marker="o", s=6, color="#7DCBFF", label="Cloud")
		plt.plot(duct_range, medians, color="b", label="Cloud, median")
		# XGB
		plt.plot(self.x_range, xgb_predictions, color="r", label="XGB")
		# EC8
		plt.plot(self.x_range, ec8, color="#85FE81", label="EC8")
		# Krawinkler
		plt.plot(self.x_range, krawinkler, color="#E62BE1", label="Krawinkler & Nassar (1992)")
		# Miranda
		plt.plot(self.x_range, miranda, color="#7F04F9", label="Miranda (1993)")
		# Newmark
		plt.plot(self.x_range, newmark, color="#D9F904", label="Newmark & Hall (1992)")
		# Vidic
		plt.plot(self.x_range, vidic, color="#F96B04", label="Vidic et al. (1994)")
		# Guerrini
		plt.plot(self.x_range, guerrini, color="#6297B5", label="Guerrini et al. (2017)")

		# Plot limits
		plt.xlim([0, 8])
		plt.ylim([0, 7])

		plt.xlabel(r"Observed ductility demand, $\mu_{dyn}$" + "\n(d)", fontsize=FONTSIZE)
		plt.ylabel(r"Static strength ratio, $R$", fontsize=FONTSIZE)
		plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
		plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
		ax.tick_params(axis='both', labelsize=FONTSIZE)
		if legend:
			plt.legend(frameon=False, loc='center left', fontsize=FONTSIZE, bbox_to_anchor=(1, 0.5))

		plt.show()
		export_figure(fig, filename=f"figuresComparisons/comparisons_{self.period}_{self.ductility}", filetype="svg")

	def r_vs_T(self, model_path, legend=False):
		Tc = self.soil[self.soil_class][0]
		Tg = 1.0
		site = "rock"
		Tcc = .4
		case = "FD"

		t_range = np.linspace(0.01, 3, 100)

		xgb_predict = []
		ec8 = []
		krawinkler = []
		miranda = []
		newmark = []
		vidic = []
		guerrini = []

		for T in t_range:
			ec8.append(EC8_2004(self.ductility, T, Tc))
			krawinkler.append(Krawinkler_Nassar_1992(self.ductility, T, self.ah*100))
			miranda.append(Miranda_1993(self.ductility, T, site, Tg))
			newmark.append(Newmark_Hall_1992(self.ductility, T, Tcc, Tc))
			vidic.append(Vidic_et_al_1994(self.ductility, T, Tc))
			guerrini.append(fsolve(Guerrini_et_al_2017, x0=np.array([10]), factor=0.1,
			                       args=(self.ductility, T, case, Tc)))

			# Xgb
			self.period = T
			prediction = self.xgb_single_prediction(model_path)
			xgb_predict.append(prediction)

		# Plotting
		fig, ax = plt.subplots(figsize=(3.0, 2.5))

		# XGB
		plt.plot(t_range, xgb_predict, color="r", label="XGB")
		# EC8
		plt.plot(t_range, ec8, color="#85FE81", label="EC8")
		# Krawinkler
		plt.plot(t_range, krawinkler, color="#E62BE1", label="Krawinkler & Nassar (1992)")
		# Miranda
		plt.plot(t_range, miranda, color="#7F04F9", label="Miranda (1993)")
		# Newmark
		plt.plot(t_range, newmark, color="#D9F904", label="Newmark & Hall (1992)")
		# Vidic
		plt.plot(t_range, vidic, color="#F96B04", label="Vidic et al. (1994)")
		# Guerrini
		plt.plot(t_range, guerrini, color="#6297B5", label="Guerrini et al. (2017)")

		# Plot limits
		plt.xlim([0.01, 3])
		plt.ylim([1, 8])

		ax.set_xscale("log")

		plt.xlabel(r"Period, $T$ [s]" + "\n(a)", fontsize=FONTSIZE)
		plt.ylabel(r"Static strength ratio, $R$", fontsize=FONTSIZE)
		plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
		plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
		ax.tick_params(axis='both', labelsize=FONTSIZE)
		if legend:
			plt.legend(frameon=False, loc='center left', fontsize=FONTSIZE, bbox_to_anchor=(1, 0.5))

		plt.show()
		export_figure(fig, filename=f"figuresComparisons/r_vs_T_{self.ductility}", filetype="svg")

	def r_vs_mu(self, model_path, legend=False):
		Tc = self.soil[self.soil_class][0]
		Tg = 1.0
		site = "rock"
		Tcc = .4
		case = "FD"

		mu_range = np.linspace(1, 8, 100)

		xgb_predict = []
		ec8 = []
		krawinkler = []
		miranda = []
		newmark = []
		vidic = []
		guerrini = []

		for ductility in mu_range:
			ec8.append(EC8_2004(ductility, self.period, Tc))
			krawinkler.append(Krawinkler_Nassar_1992(ductility, self.period, self.ah*100))
			miranda.append(Miranda_1993(ductility, self.period, site, Tg))
			newmark.append(Newmark_Hall_1992(ductility, self.period, Tcc, Tc))
			vidic.append(Vidic_et_al_1994(ductility, self.period, Tc))
			guerrini.append(fsolve(Guerrini_et_al_2017, x0=np.array([10]), factor=0.1,
			                       args=(ductility, self.period, case, Tc)))

			# Xgb
			self.ductility = ductility
			prediction = self.xgb_single_prediction(model_path)
			xgb_predict.append(prediction)

		# Plotting
		fig, ax = plt.subplots(figsize=(3.0, 2.5))

		# XGB
		plt.plot(mu_range, xgb_predict, color="r", label="XGB")
		# EC8
		plt.plot(mu_range, ec8, color="#85FE81", label="EC8")
		# Krawinkler
		plt.plot(mu_range, krawinkler, color="#E62BE1", label="Krawinkler & Nassar (1992)")
		# Miranda
		plt.plot(mu_range, miranda, color="#7F04F9", label="Miranda (1993)")
		# Newmark
		plt.plot(mu_range, newmark, color="#D9F904", label="Newmark & Hall (1992)")
		# Vidic
		plt.plot(mu_range, vidic, color="#F96B04", label="Vidic et al. (1994)")
		# Guerrini
		plt.plot(mu_range, guerrini, color="#6297B5", label="Guerrini et al. (2017)")

		# Plot limits
		plt.xlim([1, 8])
		plt.ylim([1, 8])

		plt.xlabel(r"Period, $T$ [s]" + "\n(a)", fontsize=FONTSIZE)
		plt.ylabel(r"Static strength ratio, $R$", fontsize=FONTSIZE)
		plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
		plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
		ax.tick_params(axis='both', labelsize=FONTSIZE)
		if legend:
			plt.legend(frameon=False, loc='center left', fontsize=FONTSIZE, bbox_to_anchor=(1, 0.5))

		plt.show()

		export_figure(fig, filename=f"figuresComparisons/r_vs_mu_{self.period}_", filetype="svg")


if __name__ == "__main__":

	path = Path.cwd()
	model_path = path.parents[0] / "models"

	period = 2.5
	ductility = 6.0

	ductility_dyn_range = np .arange(1.05, ductility+0.05, 0.05)

	r_mu_t = RMUTModels(ductility_dyn_range, period, ductility)
	# r_mu_t.plot_all_cloud(model_path, legend=False)
	# r_mu_t.r_vs_T(model_path)
	r_mu_t.r_vs_mu(model_path)
