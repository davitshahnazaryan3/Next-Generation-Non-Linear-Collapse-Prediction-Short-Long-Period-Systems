from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
import numpy as np
import pandas as pd


class RemoveColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns_to_remove):
		"""
		Remove undesirable columns from the dataset
		:param columns_to_remove: List[str]
		"""
		self.columns_to_remove = columns_to_remove

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		"""
		Transform the input DataFrame by iteratively removing columns specified
		:param X: DataFrame             Dataset to be transformed
		:param y: DataFrame             Target dataset
		:return: DataFrame              Transformed dataset
		"""
		for column in self.columns_to_remove:
			try:
				X.drop(column, axis=1, inplace=True)
			except LookupError:
				continue

		return X


class LabelEncoderExt(object):
	def __init__(self):
		"""
		Differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
		Unknown will be added in fit and transform will take care of new item. It gives unknown class id
		"""
		self.label_encoder = LabelEncoder()

	def fit(self, data_list):
		"""
		Fits the encoder for all unique values and introduces an unknown value
		:param data_list: List[str]
		:return: self
		"""
		self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
		self.classes_ = self.label_encoder.classes_

		return self

	def transform(self, data_list):
		"""
		Transforms the data list to id list where new values get assigned to Unknown class
		:param data_list: List[str]
		:return: List
		"""
		new_data_list = list(data_list)
		for unique_item in np.unique(data_list):
			if unique_item not in self.label_encoder.classes_:
				new_data_list = ['Unknown' if x == unique_item else x for x in new_data_list]

		return self.label_encoder.transform(new_data_list)


class LSOAEncoder(BaseEstimator, TransformerMixin):
	LSOA_encoder = LabelEncoderExt()

	def __init__(self, column_name=""):
		"""
		Initialize transformer with a column that needs to be transformed
		:param column_name: name of a column to be transformed, string
		"""
		self.column_name = column_name

	def fit(self, X, y=None):
		"""
		This will fit LabelEmcoderExt transformer on a column self.column_name = column_name
		:param X: A dataset to be transformed
		:return: self
		"""
		self.LSOA_encoder.fit(X[self.column_name].astype(str))
		return self

	def transform(self, X, y=None):
		"""
		This will transform the input self.column_name column of the dataset
		:param X: A dataset to be transformed
		:return: X
		"""
		X[self.column_name] = self.LSOA_encoder.transform(X[self.column_name].astype(str))
		return X


class NanImputer(BaseEstimator, TransformerMixin):
	def __init__(self, strategy="category"):
		"""
		Imputer for missing values. Initialize transformed with an imputing strategy to be performed
		:param strategy: str                Possible values: "category", "model"
		"""
		self.strategy = strategy

	def fit(self, X, y=None):
		"""
		If impute_strategy is "model", fit the Multivariate imputer IterativeImputer on imputing missing values
        by training BayesianRidge() that estimates each feature from all the others by
        modeling each feature with missing values as a function of other features in a
        round-robin fashion.
        :param X: DataFrame                 A dataset to be transformed
		:param y:
		:return: self
		"""
		if self.strategy == "model":
			self.imputer = IterativeImputer(BayesianRidge())
			self.imputer.fit(X)
		return self

	def transform(self, X, y=None):
		"""
		Transforms the input dataset via performing one of the implemented imputing strategies.
		"category" - create a "-1" category for missing value
		"model" - fills missing values based on the other features in the dataset
		:param X: DataFrame             A dataset to be transformed
		:return: DataFrame              Transformed dataset
		"""
		if self.strategy == "category":
			X.fillna(-1, inplace=True)
			return X
		if self.strategy == "model":
			columns = X.columns
			return pd.DataFrame(self.imputer.transform(X), columns=columns)


class RandomSamplingImputation(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		for column in self.columns:
			s = X[column].value_counts(normalize=True)
			missing = X[column].isnull()
			X.loc[missing, column] = np.random.choice(s.index, size=len(X[missing]), p=s.values)

		return X
