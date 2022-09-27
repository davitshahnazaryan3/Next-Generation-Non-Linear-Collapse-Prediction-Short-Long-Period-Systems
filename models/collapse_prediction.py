from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
import joblib
from utils import success_msg, export_results
from models.pipelines import RemoveColumns
from models.preprocessor import Preprocessor
from models.xgboostModel import XGBoostModel
import xgboost
import pandas as pd
pd.set_option('display.max_columns', None)


def create_pipelines(train, test, y_train, y_test, logtransform=True):

    scaler = MinMaxScaler()

    pipeline = Pipeline([
        ('remove_columns', RemoveColumns(VARIABLES_TO_REMOVE)),
        ('scaler', scaler)
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

    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    success_msg(f"3. Data preprocessed. Number of independent variables: {len(X_train[0])}")

    return dtrain, dtest, y_train, y_test, X_train, X_test, scaler


def run_model(X_train, y_train, X_test, y_test, method):
    if method == "knn":
        model = KNeighborsRegressor()
    elif method == "linear":
        model = LinearRegression()
    elif method == "dt":
        model = DecisionTreeRegressor()

    model.fit(X_train, y_train)

    r_sq = model.score(X_train, y_train)

    prediction = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # plt.scatter([1]*len(y_test), y_test)
    # plt.show()

    return model, mae_train, mae, r_sq, r2


if __name__ == "__main__":
    path = Path.cwd().parents[0]

    dv = "ro_3"
    ml = "xgb"

    if dv == "ro_3":
        df_name = "df_v3_rho3_collapse"
    else:
        df_name = "df_v3_rho2_collapse"

    VARIABLES_TO_REMOVE = ["identifier", "record", "damping", "R", "say", "hardening_ratio", "ductility_end",
                           "actual_ductility_end", "ro_2"]

    model = Preprocessor(path, dv, df_name, include_collapse=True)
    X, y = model.get_data()

    X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.3, random_state=1)
    print("Independent variables:")
    print("\t", X_train.columns)

    dtrain, dtest, y_train, y_test, X_train, X_test, scaler = create_pipelines(X_train, X_test, y_train, y_test)

    # # ---- DT
    # method, mae_train, mae, r_sq, r2 = run_model(X_train, y_train, X_test, y_test, ml)

    # ---- XGBoost
    model = XGBoostModel(dtrain, dtest, y_train, y_test)

    model.NUM_BOOST_ROUND = 50

    params = {'max_depth': 7, 'min_child_weight': 6, 'eta': .3, 'subsample': 1.0, 'colsample_bytree': 1.0,
              'objective': "reg:linear", 'eval_metric': "mae", 'verbosity': 0,
              "n_estimators": 100}

    method, mae_train, mae, r_sq, r2 = model.best_model_training(params)

    print(mae_train, mae, r_sq, r2)

    # Denormalize, and inverse logtransform
    predictions = method.predict(dtest)

    X_test = scaler.inverse_transform(X_test)
    y_test = np.expm1(y_test)
    predictions = np.expm1(predictions)

    # export model
    joblib.dump(method, f"{dv}_{ml}_collapse.sav")

    # Other info
    outs = {
        "method": ml,
        "dv": dv,
        "metrics": [mae_train, mae, r_sq, r2],
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
        "predictions": predictions,
    }

    export_results(f"{dv}_{ml}_collapse", outs, "pickle")

