from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from utils import success_msg, export_results
from models.preprocessor import Preprocessor
from models.xgboostModel import XGBoostModel
from models.hyperparameters import *


def train_base_model():
	# Base model
	mean_train = np.mean(np.array(y_train), axis=0)
	baseline_predictions = np.ones(y_test.shape) * mean_train

	mae_baseline = mean_absolute_error(np.array(y_test), baseline_predictions)

	mae = mae_baseline
	average = np.mean(np.array(y_test))
	division = mae_baseline / average * 100

	print("Baseline MAE is", mae)
	print("The average is", average)
	print(division)
	success_msg("The error is not too bad...\n")


if __name__ == "__main__":
	path = Path.cwd()

	# Pre-processing
	dv = "R"

	if dv == "ro_3":
		df_name = "df_v3_rho3"
	else:
		df_name = "df_v3_rho2"

	model = Preprocessor(path, dv, df_name)
	X, y = model.get_data()
	X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.3, random_state=1)

	dtrain, dtest, y_train, y_test, X_train, X_test = model.create_pipelines(X_train, X_test, y_train, y_test,
	                                                                         min_max_scaling=True, logtransform=True)

	# Model training
	model = XGBoostModel(dtrain, dtest, y_train, y_test)

	# Tuning max_depth and min_child_weight
	params = {'max_depth': 6, 'min_child_weight': 1, 'eta': .3, 'subsample': 1, 'colsample_bytree': 1,
	          'objective': "reg:squaredlogerror", 'eval_metric': "mae", 'verbosity': 0,
	          "n_estimators": 100}

	# # Tune all hyperparameters
	# model = XGBoostModel(X_train, X_test, y_train, y_test)
	# model.run_optimization_algorithm()

	# grid = get_max_depth_min_child_weight()
	# model.hyperparameter_tuning(params, grid, ["max_depth", "min_child_weight"], "params_max_depth_min_child_weight")

	# grid = get_eta()
	# model.single_hyperparameter_tuning(params, grid, ["eta"], "params_eta")

	# # reg_alpha, reg_lambda
	# grid = get_regularization_parameters()
	# model.hyperparameter_tuning(params, grid, ["reg_alpha", "reg_lambda"], "regularization_params")

	# ------------------- Best Model fitting
	model.NUM_BOOST_ROUND = 100

	params = {'max_depth': 7, 'min_child_weight': 6, 'eta': .3, 'subsample': 1.0, 'colsample_bytree': 1.0,
	          'objective': "reg:linear", 'eval_metric': "mae", 'verbosity': 0,
	          "n_estimators": 100}

	best_model, mae_train, mae, r_sq, r2 = model.best_model_training(params)

	# export model
	joblib.dump(best_model, f"{dv}_xgb.sav")

	# Other info
	outs = {
	    "method": "xgb",
	    "dv": dv,
	    "metrics": [mae_train, mae, r_sq, r2],
	    "y_train": y_train,
	    "y_test": y_test,
	    "X_train": X_train,
	    "X_test": X_test,
	}

	export_results(f"{dv}_xgb", outs, "pickle")

