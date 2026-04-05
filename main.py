import pandas as pd
import numpy as np
import matplotlib as mp
import EDA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor


def categoric_trans(df):
    df = pd.get_dummies(df, columns=["Model", "Region", "Color", "Fuel Type", "Turbo"])
    return df


def train_test(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Base Price (USD)"]), df["Base Price (USD)"], test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, depth=5, learning_rate=0.01, n_estimators=1000):
    model = CatBoostRegressor(
        depth=5,
        learning_rate=0.01,
        n_estimators=1000,
        random_state=42,
        verbose=100,
        #cat_features=["Model", "Region", "Color", "Fuel Type", "Turbo"]
        )

    random_forest = RandomForestRegressor(
        n_estimators=550,
        random_state=42,
        verbose=100)

    random_forest.fit(X_train, y_train)
    model.fit(X_train, y_train)

    return random_forest, model


def predict(model, X_test):
    return model.predict(X_test)


def sings_importance(model, columns, model_name: str):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    cols = np.array(columns)[order]
    vals = imp[order]
    plt.figure(figsize=(10, max(6, 0.25 * len(cols))))
    plt.barh(cols[::-1], vals[::-1])  # сверху вниз: от большего к меньшему
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Важность признаков — {model_name}")
    plt.tight_layout()
    plt.show()


def main(df, depth=5, learning_rate=0.01, n_estimators=1000):
    X_train, X_test, y_train, y_test = train_test(categoric_trans(df))
    random_forest, model = train_model(X_train, y_train, depth=depth, learning_rate=learning_rate, n_estimators=n_estimators)
    y_pred_catboost = predict(model, X_test)
    y_pred_random_forest = predict(random_forest, X_test)
    print("Модели обучились!")
    return y_test, y_pred_catboost, y_pred_random_forest, random_forest, model, X_test.columns


if __name__ == "__main__":
    df = pd.read_csv("mercedes_benz_sales_2020_2025_sample_10000.csv")
    eda = EDA.EDA(df, "Base Price (USD)")

    eda.label()
    eda.pyplot()
    eda.corr_coef()
    
    y_test, pred_catboost, pred_random_forest, random_forest, model, columns = main(df)
    
    print(f"R2 RandomForest: {r2_score(y_test, pred_random_forest)}")
    print(f"MAE RandomForest: {mean_absolute_error(y_test, pred_random_forest)}")
    print("--------------------------------")
    print(f"R2 CatBoost: {r2_score(y_test, pred_catboost)}")
    print(f"MAE CatBoost: {mean_absolute_error(y_test, pred_catboost)}")

    sings_importance(random_forest, columns, "RandomForestRegressor")
    sings_importance(model, columns, "CatBoostRegressor")

    res = pd.DataFrame({"y_test": y_test, "pred_catboost": pred_catboost, "pred_random_forest": pred_random_forest, "error_catboost": np.abs(y_test - pred_catboost), "error_random_forest": np.abs(y_test - pred_random_forest)})
    print(res.head(10))

    eda.pred_vs_target(y_test, pred_catboost)
    eda.pred_vs_target(y_test, pred_random_forest)
