import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from utils import export_results, initiate_msg, success_msg, error_msg, get_init_time, get_time


class XGBoostModel:
	NUM_BOOST_ROUND = 100
	EARLY_STOPPING_ROUNDS = 10

	SPACE = {
		'max_depth': hp.quniform("max_depth", 3, 18, 1),
		'gamma': hp.uniform('gamma', 1, 9),
		'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
		'reg_lambda': hp.uniform('reg_lambda', 0, 1),
		'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
		'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
		'n_estimators': 180,
		'seed': 1,
	}

	# Parameters to be tuned, objective function, evaluation metric and verbosity
	PARAMS = {
		'max_depth': 6,
		'min_child_weight': 1,
		'eta': .3,
		'subsample': 1,
		'colsample_bytree': 1,

		# Other parameters
		'objective': "reg:linear",
		'eval_metric': "mae",
		'verbosity': 0
	}

	def __init__(self, dtrain, dtest, y_train, y_test, run_baseline_model=False):
		self.dtrain = dtrain
		self.dtest = dtest
		self.y_train = y_train
		self.y_test = y_test

		# Run initial model training
		if run_baseline_model:
			self._baseline_model_training()

	def tune_all(self, space):
		model = xgb.XGBRegressor(
			n_estimators=space["n_estimators"],
			max_depth=int(space["max_depth"]),
			gamma=space["gamma"],
			reg_alpha=int(space["reg_alpha"]),
			min_child_weight=int(space["min_child_weight"]),
			colsample_bytree=int(space["colsample_bytree"])
		)

		evaluation = [(self.dtrain, self.y_train), (self.dtest, self.y_test)]

		model.fit(self.dtrain, self.y_train, eval_set=evaluation,
		          eval_metric="mae", early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
		          verbose=False)

		predictions = model.predict(self.dtest)
		mae = mean_absolute_error(self.y_test, predictions)
		r2 = r2_score(self.y_test, predictions)

		print("MAE: ", mae, "R2: ", r2)

		return {"loss": mae, "status": STATUS_OK}

	def run_optimization_algorithm(self):
		trials = Trials()

		best_hyperparams = fmin(fn=self.tune_all,
		                        space=self.SPACE,
		                        algo=tpe.suggest,
		                        max_evals=100,
		                        trials=trials)

		success_msg("The best hyperparameters are : ")
		print(best_hyperparams)
		return best_hyperparams

	def _baseline_model_training(self, seed=1, nfold=5):
		initiate_msg("4. Baseline XGBoost model training...")

		start = get_init_time()

		model = xgb.cv(
			self.PARAMS,
			self.dtrain,
			num_boost_round=self.NUM_BOOST_ROUND,
			seed=seed,
			nfold=nfold,
			metrics={self.PARAMS["eval_metric"]},
			early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
		)

		success_msg("Minimum test MAE on average: {:.2f}".format(
			model["test-mae-mean"].min()))

		get_time(start)

	def hyperparameter_tuning(self, params, grid, params_to_tune, filename, seed=1, nfold=5):
		min_mae = float("inf")
		best_params = None

		start = get_init_time()

		for param1, param2 in grid:
			print("CV with {}={}, {}={}".format(params_to_tune[0],
			                                    param1, param2,
			                                    params_to_tune[1]))

			# update parameters
			params[params_to_tune[0]] = param1
			params[params_to_tune[1]] = param2

			# Run CV
			cv = xgb.cv(
				params,
				self.dtrain,
				num_boost_round=self.NUM_BOOST_ROUND,
				seed=seed,
				nfold=nfold,
				metrics={'mae'},
				early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
			)

			# Update best MAE
			mean_mae = cv['test-mae-mean'].min()
			boost_rounds = cv['test-mae-mean'].argmin()
			print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
			if mean_mae < min_mae:
				min_mae = mean_mae
				best_params = (param1, param2)

		success_msg("Best params: {} = {}, {} = {}, MAE: {}".format(params_to_tune[0], best_params[0],
		                                                            params_to_tune[1], best_params[1],
		                                                            min_mae))

		params[params_to_tune[0]] = best_params[0]
		params[params_to_tune[1]] = best_params[1]

		get_time(start)

		print("Exporting outputs")
		export_results(f"models/{filename}", params, "json")

	def single_hyperparameter_tuning(self, params, grid, params_to_tune, filename, seed=1, nfold=5):
		min_mae = float("inf")
		best_params = None

		start = get_init_time()

		for param1 in grid:
			print("CV with {}={}".format(params_to_tune[0], param1))

			# update parameters
			params[params_to_tune[0]] = param1

			# Run CV
			cv = xgb.cv(
				params,
				self.dtrain,
				num_boost_round=self.NUM_BOOST_ROUND,
				seed=seed,
				nfold=nfold,
				metrics={'mae'},
				early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
			)

			# Update best MAE
			mean_mae = cv['test-mae-mean'].min()
			boost_rounds = cv['test-mae-mean'].argmin()
			print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
			if mean_mae < min_mae:
				min_mae = mean_mae
				best_params = param1

		success_msg("Best params: {} = {}, MAE: {}".format(params_to_tune[0], best_params, min_mae))

		params[params_to_tune[0]] = best_params

		get_time(start)

		print("Exporting outputs")
		export_results(f"models/{filename}", params, "json")

	def best_model_training(self, params):

		watchlist = [(self.dtrain, 'train'), (self.dtest, 'eval')]
		progress = dict()

		best_model = xgb.train(
			params,
			self.dtrain,
			self.NUM_BOOST_ROUND,
			watchlist,
			evals=[(self.dtest, "Test")],
			evals_result=progress,
		)

		train_predict = best_model.predict(self.dtrain)
		test_predict = best_model.predict(self.dtest)
		mae = mean_absolute_error(test_predict, self.y_test)

		success_msg("{}".format(mae))
		r2 = r2_score(self.y_test, test_predict)
		r2_train = r2_score(self.y_train, train_predict)

		mae_train = mean_absolute_error(train_predict, self.y_train)

		print(mae_train)
		print("Train R2 score", r2_train)
		print("Test R2 score {}".format(r2))

		# print(self.y_test)
		# print(test_predict)

		return best_model, mae_train, mae, r2_train, r2
