import os
import pickle
import pandas as pd
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from xgboostModel.pipelines import RemoveColumns
from utils import export_results, initiate_msg, success_msg, error_msg


class EntryDb:
	def __init__(self, df, entry):
		self.df = pd.concat([
			df, entry
		], ignore_index=True)


class Preprocessor:
	VARIABLES_TO_REMOVE = ["identifier", "record", "damping", "ductility_end", "say"]

	def __init__(self, path, dependent_variable, dataframe_filename, include_collapse=False):
		"""
		DataFrame columns:
			identifier: int                 Unique identifier
			record: str                     Record name
			period: float                   Period of the SDOF, [s]
			damping: float
			R: float
			hardening_ratio: float          Hardening slope (not ratio)
			ductility: float                Hardening ductility
			ductility_end: float            Fracturing ductility
			actual_ductility_end: float     Computed fracturing ductility
			ro_2: float                     Sa_avg_2T / Say
			ro_3: float                     Sa_avg_3T / Say

		Independent variables to be dropped:
			identifier, record

		Dependent variables:
			ro_2 for first model
			ro_3 for second model

		Multi-variate regression models

		:param path: Path
		:param dependent_variable: str
		:param dataframe_filename: str
		"""

		if dependent_variable == "ro_2":
			self.VARIABLES_TO_REMOVE.extend(["ro_3", "R"])
		elif dependent_variable == "ro_3":
			self.VARIABLES_TO_REMOVE.extend(["ro_2", "R"])
		else:
			self.VARIABLES_TO_REMOVE.extend(["ro_2", "ro_3"])

		self.path = path
		self.dependent_variable = dependent_variable
		self.include_collapse = include_collapse

		initiate_msg(f"1. Reading dataframe")

		self.df = self.create_dataframe(dataframe_filename)

		success_msg(f"1. Read cached DataFrame. Number of data points: {len(self.df.index)}")

	def create_dataframe(self, dataframe_filename):
		if os.path.exists(self.path / "data" / (dataframe_filename + ".csv")):
			return pd.read_csv(self.path / "data" / (dataframe_filename + ".csv"))

		# Currently including Say
		df = pd.DataFrame(columns=["identifier", "record", "period", "damping", "R", "say", "hardening_ratio",
		                           "ductility", "ductility_end", "actual_ductility_end",
		                           "ro_2", "ro_3"])

		# TODO, data here
		path = self.path.parents[0] / "recordSelection/database/df_rho3_v3"

		total_length = 0
		for filename in os.listdir(path):
			entry = pd.read_csv(path / filename, index_col=0)
			total_length += len(entry)

			if not self.include_collapse:
				entry = entry[entry["actual_ductility_end"] <= entry["ductility_end"]]
			else:
				# Select only collapsed cases
				entry = entry[entry["actual_ductility_end"] > entry["ductility_end"]]

			entrydb = EntryDb(df, entry)
			df = entrydb.df

		# path = self.path.parents[0] / "recordSelection/database/df_v1"
		#
		# for filename in os.listdir(path):
		# 	entry = pd.read_csv(path / filename, index_col=0)
		# 	entry = entry[entry["actual_ductility_end"] <= entry["ductility_end"]]
		# 	entrydb = EntryDb(df, entry)
		# 	df = entrydb.df

		# Into a single global dataframe
		self._export_db(df, dataframe_filename)

		print("Total length, collapse+non-collapse", total_length)

		return df

	def _export_db(self, df, dataframe_filename):
		filename = self.path / "data" / dataframe_filename

		export_results(filename, df, "csv")

	def get_data(self):
		self.df = self.df.sample(frac=1)

		# Select only records pertinent to target_variable
		# if self.dependent_variable == "ro_3":
		# 	records = json.load(open(self.path.parents[0] / "recordSelection/recordSelection/data/data.json"))
		#
		# 	for period in ["2.5"]:
		# 		rsn = records[period]['3']['rsn']
		# 		filenames = records[period]['3']['filenames']
		#
		# 		record_set = []
		# 		for i in range(len(rsn)):
		# 			record_set.append(f"RSN{rsn[i]}_{filenames[i]}")
		#
		# 		print(len(self.df))
		# 		print(period)
		# 		self.df = self.df.drop(self.df[(~self.df["record"].isin(record_set)) &
		# 		                               (self.df["period"] == float(period))].index)
		# 		print(len(self.df))
		#
		# 		exit()

		X = self.df.drop(self.dependent_variable, axis=1)
		y = self.df[self.dependent_variable]

		# pd.set_option('display.max_columns', None)
		# print(y.describe())
		# exit()
		return X, y

	def train_test_split(self, X, y, test_size=0.2, random_state=1, path=None):

		initiate_msg(f"2. Preparing train test split of data with test size: {test_size}")

		# TODO filename
		pickle_file = "a"

		if path is None:
			path = self.path / f"data/{pickle_file}.pickle"

		if os.path.exists(path):
			with open(path, "rb") as file:
				data = pickle.load(file)

				X_train = data["X_train"]
				X_test = data["X_test"]
				y_train = data["y_train"]
				y_test = data["y_test"]

			success_msg("2. Read cached train test split data")

			return X_train, X_test, y_train, y_test

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

		split_data = {
			"X_train": X_train,
			"X_test": X_test,
			"y_train": y_train,
			"y_test": y_test
		}

		# TODO
		# export_results(self.path / f"data/{pickle_file}", split_data, "pickle")

		success_msg("2. Train test split complete")

		return X_train, X_test, y_train, y_test

	def create_pipelines(self, train, test, y_train, y_test, min_max_scaling=True, logtransform=False):
		"""
		Preprocessing steps before data split
			1. Removal of unnecessary independent variables
			2. Removal of irrelevant dependent variable (separate models for ro_2 and ro_3)
			3. Min max scaling of all variables

		:param train: DataFrame
		:param test: DataFrame
		:param y_train: Series
		:param y_test: Series
		:param min_max_scaling: bool
		:param logtransform: bool
		:return: DMatrix, DMatrix
		"""

		initiate_msg(f"3. Preparing data for preprocessing through pipelines")
		print("Variables to remove: ")
		print("\t", self.VARIABLES_TO_REMOVE)

		if min_max_scaling:
			pipeline = Pipeline([
				('remove_columns', RemoveColumns(self.VARIABLES_TO_REMOVE)),
				('scaler', MinMaxScaler())
			])
		else:
			pipeline = Pipeline([
				('remove_columns', RemoveColumns(self.VARIABLES_TO_REMOVE)),
			])

		if logtransform:
			pipeline_target = Pipeline([
				("log_transform", FunctionTransformer(np.log1p, validate=True))
			])

			y_train = np.array(y_train)
			y_test = np.array(y_test)

			y_train = y_train.reshape(-1, 1)
			y_test = y_test.reshape(-1, 1)

			y_train = pipeline_target.fit_transform(y_train)
			y_test = pipeline_target.transform(y_test)

		X_train = pipeline.fit_transform(train)
		X_test = pipeline.transform(test)

		dtrain = xgb.DMatrix(X_train, label=y_train)
		dtest = xgb.DMatrix(X_test, label=y_test)

		success_msg(f"3. Data preprocessed. Number of independent variables: {len(X_train[0])}")

		return dtrain, dtest, y_train, y_test, X_train, X_test
