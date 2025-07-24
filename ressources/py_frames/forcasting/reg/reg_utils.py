import pandas as pd


def fill_gaps_(data, date_min, date_max):
    # 1) Copie de sécurité
    client_id = None
    if isinstance(data, pd.DataFrame):
        df_local = data.copy()
    if isinstance(data, tuple):
        df_local = data[1].copy()
        client_id = data[0]
    else:  # Assuming data is a DataFrame if not a tuple
        df_local = data.copy()
    # 2) S'assurer que la colonne date est bien en datetime
    if "date" not in df_local.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'date'.")
    df_local["date"] = pd.to_datetime(df_local["date"])

    # 3) Calcul de la date_min
    if date_min is None:
        date_min = df_local["date"].min()
    if date_max is None:
        date_max = df_local["date"].max()

    date_min = pd.to_datetime(date_min)
    date_max = pd.to_datetime(date_max)

    # 5) Générer la plage de dates mensuelles (du 1er de chaque mois)
    all_months = pd.date_range(start=date_min, end=date_max, freq="MS")

    # 6) Créer un DataFrame indexé par ces mois
    df_all = pd.DataFrame({"date": all_months})

    # 7) Joindre avec df_local (merge ou reindex). On fait un left join
    #    pour garder toutes les lignes de df_all.
    df_merged = pd.merge(df_all, df_local, on="date", how="left")

    # 8) Optionnel : combler les valeurs manquantes
    # (par exemple, 0 transactions, 0 montants, etc.)
    # selon vos colonnes. Ex :
    cols_to_fill = [
        col
        for col in df_merged.columns
        if "nb_virements" in col or "mnt_virements" in col
    ]
    for col in cols_to_fill:
        df_merged[col] = df_merged[col].fillna(0)

    # 9) Tri par date (normalement déjà, mais au cas où)
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    df_merged["client"] = df_local["client"].iloc[0]

    if "code_guichet" in df_merged.columns:
        mode_guichet = df_local["code_guichet"].mode()
        if not mode_guichet.empty:
            df_merged["code_guichet"] = mode_guichet[0]

    return df_merged


# df = pd.read_pickle("data_final.pkl")
# df_clients = df.groupby(by='client')
# min_date = df['date'].min()
# max_date = df['date'].max()
# print(min_date, max_date)
# dfs = []
# for i, client in enumerate(df_clients):
#     client = fill_gaps(data=client, date_min=min_date, date_max=max_date)
#     dfs.append(client)
#     if i%1000 == 0:
#         print(f"{i}/19104")
#         # break

# print(len(dfs))
# df_clients_filled = pd.concat(dfs)
# df_clients_filled
# df_clients_filled.to_pickle("data_filled_2023-01-01_2025-02-01.pkl")

import numpy as np
import pandas as pd


def get_data(df, num=100):

    df = df.copy()

    ratio_cols_nb = df.filter(regex="^nb_").columns
    ratio_cols_mnt = df.filter(regex="^mnt_").columns

    df_ratio = df.copy()

    # 1) Totaux par ligne
    tot_nb = df_ratio[ratio_cols_nb].sum(axis=1)
    tot_mnt = df_ratio[ratio_cols_mnt].sum(axis=1)

    # 2) Ratios (évite division 0)
    df_ratio.loc[tot_nb > 0, ratio_cols_nb] = df_ratio.loc[
        tot_nb > 0, ratio_cols_nb
    ].div(tot_nb[tot_nb > 0], axis=0)
    df_ratio.loc[tot_mnt > 0, ratio_cols_mnt] = df_ratio.loc[
        tot_mnt > 0, ratio_cols_mnt
    ].div(tot_mnt[tot_mnt > 0], axis=0)

    # 3) Canal préféré PAR LIGNE (seulement si tot_nb > 0)
    mask_active = tot_nb > 0
    pref_codes = (
        df_ratio.loc[mask_active, ratio_cols_nb]
        .idxmax(axis=1, skipna=True)  # jamais all-NA sous mask
        .str.replace("nb_virements_", "", regex=False)
    )

    df_ratio["preferred_channel"] = "inactif"  # valeur par défaut
    df_ratio.loc[mask_active, "preferred_channel"] = pref_codes

    # 4) Insertion dans le DF final
    df["preferred_channel"] = df_ratio["preferred_channel"]

    # print(df[['date', 'client', 'preferred_channel']].head())
    # print(df['preferred_channel'].value_counts(dropna=False))

    if num == -1:
        return df, df_ratio
    unique_clients_all = df["client"].unique()
    clients_to_select = unique_clients_all[:num]
    df_ratio = df_ratio[df_ratio["client"].isin(clients_to_select)]
    df = df[df["client"].isin(clients_to_select)]

    return df, df_ratio


# df,_ = get_data(num=1000)
# df.to_csv("df.csv")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# import a variety of regressors
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# 0) MODEL REGISTRY
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "adaboost": AdaBoostRegressor,
    "ridge": Ridge,
    "lasso": Lasso,
    "knn": KNeighborsRegressor,
    "svr": SVR,
}


# 1) BUILD PIPELINE
def build_pipeline(numeric_feats, model_name="random_forest", model_params=None):
    """
    numeric_feats: list of feature column names to scale
    model_name: key in MODEL_REGISTRY
    model_params: dict of parameters to pass to the regressor
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    model_params = model_params or {}
    base_cls = MODEL_REGISTRY[model_name]
    estimator = base_cls(**model_params)
    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_feats),
            (
                "client",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["client"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", preproc), ("rf", MultiOutputRegressor(estimator))])


# 2) SUPERVISED DATA GENERATION
def make_supervised(
    df, channels, client_col="client", date_col="date", n_lags=6, horizon=3
):
    X, y, dates = [], [], []
    for cid, grp in df.groupby(client_col):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        arr = grp[channels].values
        for i in range(n_lags, len(arr) - horizon + 1):
            X.append(np.r_[arr[i - n_lags : i].ravel(), cid])
            y.append(arr[i : i + horizon].ravel())
            dates.append(grp.loc[i + horizon - 1, date_col])
    in_cols = [f"{ch}_lag{j+1}" for j in range(n_lags) for ch in channels] + [
        client_col
    ]
    out_cols = [f"{ch}_t+{j+1}" for j in range(horizon) for ch in channels]
    return (
        pd.DataFrame(X, columns=in_cols),
        pd.DataFrame(y, columns=out_cols),
        pd.Series(dates, name=date_col),
        in_cols,
        out_cols,
    )


# 3) DATA SPLITTING (optional hold-out)
def prepare_and_split(df, channels, n_lags=6, horizon=3, test_months=None):
    X, y, dates, in_cols, out_cols = make_supervised(
        df, channels, n_lags=n_lags, horizon=horizon
    )
    X.index = dates
    y.index = dates
    if not test_months:
        return X, None, y, None, None, in_cols, out_cols
    split_date = dates.max() - pd.DateOffset(months=test_months)
    mask = X.index <= split_date
    return (
        X.loc[mask],
        X.loc[~mask],
        y.loc[mask],
        y.loc[~mask],
        split_date,
        in_cols,
        out_cols,
    )


# 4) TRAINING
def train_model(
    X_train, y_train, in_cols, model_name="random_forest", model_params=None
):
    numeric_feats = [c for c in in_cols if c != "client"]
    pipeline = build_pipeline(numeric_feats, model_name, model_params)
    pipeline.fit(X_train, y_train)
    return pipeline


# 5) PREDICTION
def predict_model(pipeline, X_test, out_cols):
    if X_test is None:
        return pd.DataFrame([], columns=out_cols)
    y_pred = pipeline.predict(X_test)
    return pd.DataFrame(y_pred, index=X_test.index, columns=out_cols)


# 6) EVALUATION
def evaluate_model(pipeline, X_test, y_test, out_cols):
    if X_test is None or y_test is None:
        return pd.DataFrame([], columns=["MAE", "MAPE%"])
    preds = predict_model(pipeline, X_test, out_cols)
    metrics = {}
    for col in out_cols:
        mae = mean_absolute_error(y_test[col], preds[col])
        true_nz = y_test[col].replace(0, np.nan)
        mape = np.mean(np.abs((preds[col] - y_test[col]) / true_nz)) * 100
        metrics[col] = {"MAE": mae, "MAPE%": mape}
    return pd.DataFrame(metrics).T


# 7) FORECAST FUTURE
def forecast_client(
    pipeline,
    df,
    client_id,
    channels,
    in_cols,
    n_lags=6,
    horizon=3,
    date_col="date",
    client_col="client",
):
    df_c = df[df[client_col] == client_id].sort_values(date_col)
    arr = df_c[channels].values
    if len(arr) < n_lags:
        raise ValueError(
            f"Only {len(arr)} records for client {client_id}, need >= {n_lags}"
        )
    last_date = df_c[date_col].max()
    flat = arr[-n_lags:].flatten()
    row = np.concatenate([flat, [client_id]])
    X_in = pd.DataFrame({col: [val] for col, val in zip(in_cols, row)})
    y_flat = pipeline.predict(X_in)[0]
    y = y_flat.reshape(horizon, len(channels))
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    return pd.DataFrame(y, index=future_dates, columns=channels)


# 8) PLOTTING
def plot_client_history(
    client_id, df, split_date, X_test, y_test, y_pred_df, channels, horizon
):
    hist = df[df["client"] == client_id].sort_values("date")
    print(f" hist shape {hist.shape}")
    if split_date is not None:
        hist = hist[hist["date"] <= split_date]
    if X_test is not None:
        mask = X_test["client"] == client_id
        dts = X_test.index[mask]
        yt = y_test.loc[mask]
        yp = y_pred_df.loc[mask]
    else:
        dts, yt, yp = [], pd.DataFrame(), pd.DataFrame()
    for ch in channels:
        plt.figure(figsize=(8, 3))
        plt.plot(hist["date"], hist[ch], c="blue", label="History")
        if not yt.empty and not yp.empty:
            for j in range(1, horizon + 1):  # Iterate up to the forecast horizon
                plt.plot(
                    dts,
                    yt[f"{ch}_t+1"],
                    marker="o",
                    linestyle="-",
                    label=f"True",
                )
                plt.plot(
                    dts,
                    yp[f"{ch}_t+1"],
                    marker="x",
                    linestyle="--",
                    label=f"Pred",
                )
        plt.title(f"Client {client_id}")
        plt.xlabel("Date")
        plt.ylabel(ch)
        plt.legend()
        plt.tight_layout()
        plt.show()


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    channels = [
        "nb_virements_adria_web_bo",
        "nb_virements_agence_physique",
        "nb_virements_autre_digital",
        "nb_virements_ebics",
        "nb_virements_rpa",
    ]
    df = pd.read_pickle(
        "C:/Users/Hp/Desktop/production/nwe_approach/app/ressources/py_frames/forcasting/forecast/data.pkl"
    )
    df, _ = get_data(df, num=30)
    df["date"] = pd.to_datetime(df["date"])

    # split or train on all
    X_tr, X_te, y_tr, y_te, split_date, in_cols, out_cols = prepare_and_split(
        df, channels, n_lags=6, horizon=3, test_months=6
    )

    # choose your model
    pipeline = train_model(
        X_tr,
        y_tr,
        in_cols,
        model_name="gradient_boosting",
        model_params={"n_estimators": 50, "learning_rate": 0.1},
    )

    eval_df = evaluate_model(pipeline, X_te, y_te, out_cols)
    print("Eval:\n", eval_df)

    preds_df = predict_model(pipeline, X_te, out_cols)
    plot_client_history(
        client_id=5200000007369,
        df=df,
        split_date=split_date,
        X_test=X_te,
        y_test=y_te,
        y_pred_df=preds_df,
        channels=channels,
        horizon=3,
    )

    fc = forecast_client(
        pipeline,
        df,
        client_id=5200000007369,
        channels=channels,
        in_cols=in_cols,
        n_lags=6,
        horizon=3,
    )
    print("Forecast:\n", fc)
