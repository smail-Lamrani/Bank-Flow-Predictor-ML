import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier
from dataclasses import dataclass, field
from typing import Dict, List, Any
from tqdm import tqdm

import pandas as pd
import numpy as np

def get_data(path, num = 100):

    df = pd.read_pickle(path)
    # df = df.copy()

    ratio_cols_nb  = df.filter(regex='^nb_').columns
    ratio_cols_mnt = df.filter(regex='^mnt_').columns

    df_ratio = df.copy()

    # 1) Totaux par ligne
    tot_nb  = df_ratio[ratio_cols_nb ].sum(axis=1)
    tot_mnt = df_ratio[ratio_cols_mnt].sum(axis=1)

    # 2) Ratios (évite division 0)
    df_ratio.loc[tot_nb  > 0, ratio_cols_nb ]  = df_ratio.loc[tot_nb  > 0, ratio_cols_nb ].div(tot_nb [tot_nb  > 0], axis=0)
    df_ratio.loc[tot_mnt > 0, ratio_cols_mnt] = df_ratio.loc[tot_mnt > 0, ratio_cols_mnt].div(tot_mnt[tot_mnt > 0], axis=0)

    # 3) Canal préféré PAR LIGNE (seulement si tot_nb > 0)
    mask_active = tot_nb > 0
    pref_codes  = (
        df_ratio.loc[mask_active, ratio_cols_nb]
                .idxmax(axis=1, skipna=True)         # jamais all-NA sous mask
                .str.replace('nb_virements_', '', regex=False)
    )

    df_ratio['preferred_channel'] = 'inactif'        # valeur par défaut
    df_ratio.loc[mask_active, 'preferred_channel'] = pref_codes

    # 4) Insertion dans le DF final
    df['preferred_channel'] = df_ratio['preferred_channel']

    # print(df[['date', 'client', 'preferred_channel']].head())
    # print(df['preferred_channel'].value_counts(dropna=False))

    if num == -1:
        return df, df_ratio
    unique_clients_all = df['client'].unique()
    clients_to_select = unique_clients_all[:num]
    df_ratio = df_ratio[df_ratio['client'].isin(clients_to_select)]
    df = df[df['client'].isin(clients_to_select)]

    return df, df_ratio

import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier
from dataclasses import dataclass, field
from typing import Dict, List, Any
from tqdm import tqdm

# Classifier mapping
CLASSIFIERS = {
    'logistic': LogisticRegression,
    'random_forest': RandomForestClassifier,
    'extra_trees': ExtraTreesClassifier,
    'gb': GradientBoostingClassifier,
    'hist_gb': HistGradientBoostingClassifier,
    'adaboost': AdaBoostClassifier,
    'svm': SVC,
    'knn': KNeighborsClassifier,
    'nb': GaussianNB,
    'mlp': MLPClassifier,
    'xgb': XGBClassifier
}

@dataclass
class BestParams:
    logistic: Dict[str, Any] = field(default_factory=lambda: {'C': 0.1, 'max_iter': 300, 'solver': 'lbfgs'})
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100, 'max_depth': 2, 'max_features': 'sqrt',
        'min_samples_split': 5, 'min_samples_leaf': 3, 'bootstrap': True, 'random_state': 42})
    extra_trees: Dict[str, Any] = field(default_factory=lambda: {
        'bootstrap': True, 'max_depth': 5, 'max_features': 'log2',
        'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300})
    gb: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.01, 'max_depth': 3, 'max_features': 'sqrt',
        'n_estimators': 100, 'subsample': 1.0})
    hist_gb: Dict[str, Any] = field(default_factory=lambda: {
        'l2_regularization': 1.0, 'learning_rate': 0.01, 'max_depth': None, 'max_iter': 200})
    adaboost: Dict[str, Any] = field(default_factory=lambda: {
        'algorithm': 'SAMME', 'learning_rate': 1.0, 'n_estimators': 50})
    svm: Dict[str, Any] = field(default_factory=lambda: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True})
    knn: Dict[str, Any] = field(default_factory=lambda: {'n_neighbors': 7, 'p': 1, 'weights': 'uniform'})
    nb: Dict[str, Any] = field(default_factory=lambda: {'var_smoothing': 1e-09})
    mlp: Dict[str, Any] = field(default_factory=lambda: {
        'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (50,),
        'learning_rate_init': 0.01, 'max_iter': 300})
    xgb: Dict[str, Any] = field(default_factory=lambda: {
        'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5,
        'n_estimators': 100, 'subsample': 0.8})


def add_targets_and_lags(df: pd.DataFrame, lag_months: list[int]) -> pd.DataFrame:
    df['channel_next'] = df.groupby('client')['preferred_channel'].shift(-1)
    df_clf = df.dropna(subset=['channel_next']).copy()
    ratio_cols = df_clf.filter(regex='^(nb_|mnt_)').columns
    for col in ratio_cols:
        for lag in lag_months:
            df_clf[f"{col}_lag{lag}"] = df_clf.groupby('client')[col].shift(lag).fillna(0)
    return df_clf

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['nb_switch_3m'] = (
        df.groupby('client')['preferred_channel']
          .transform(lambda s: s.ne(s.shift()).rolling(3).sum().shift(1).fillna(0))
          .astype(int)
    )
    return pd.get_dummies(df, columns=['preferred_channel'], prefix='chan_t')

def split_data(df: pd.DataFrame, features: list[str], test_frac =5):
    if test_frac == 0:
        return df[features], df['channel_next'], None, None, None, None

    train_parts = []
    test_parts = []

    for _, group in df.groupby('client'):
        group = group.sort_values('date')
        n_rows = len(group)
        n_test = test_frac
        test = group.tail(n_test)
        train = group.head(n_rows - n_test)

        test_parts.append(test)
        train_parts.append(train)

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    return train_df[features], train_df['channel_next'], test_df[features], test_df['channel_next'], train_df, test_df

def evaluate_and_plot(clf, X_test, y_test, le):
    start_time_seconds = time.perf_counter()
    y_pred = clf.predict(X_test)
    inference_duration = (time.perf_counter() - start_time_seconds) / 60
    classification_rep = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    pct_mae = (mae / (len(le.classes_) - 1)) * 100
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right')
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.show()

    return classification_rep, {'mae': mae, 'pct_mae': pct_mae}, cm, inference_duration

def predict_next_channel_for_client(pipeline, df, features, client_id, lag_months):
    # recompute lags & features on full df
    df_feat, _, le = prepare_data(df, lag_months=lag_months)
    # filter to that client
    client_rows = df_feat[df_feat['client'] == client_id]
    if client_rows.empty:
        raise ValueError(f"No data for client {client_id}")
    # pick the last available row
    X_new = client_rows[features].iloc[[-1]]
    print(X_new.columns)
    # predict
    proba = pipeline.predict_proba(X_new)
    classes = le.inverse_transform(np.arange(len(le.classes_)))
    channel_probs = dict(zip(classes, proba[0]))
    return channel_probs

def prepare_data(df, lag_months, extra_features = ['code_guichet', 'month_sin', 'month_cos', 'nb_switch_3m']):
    df = df.copy()
    df_lags = add_targets_and_lags(df, lag_months=lag_months)
    df_feat = engineer_features(df_lags)
    features = [
        *df_feat.filter(regex='^(nb_|mnt_).*_lag[123]$').columns.tolist(),
        *df_feat.filter(regex='^chan_t_').columns.tolist()] + extra_features  
     
    le = LabelEncoder().fit(df_feat['channel_next'])
    df_feat['channel_next'] = le.transform(df_feat['channel_next'])
    return df_feat, features, le

def train_classifier(X_train, y_train, model_name: str = 'gb', best_params: dict = None):
    pipeline = Pipeline([
        ('clf', CLASSIFIERS[model_name](**best_params))
    ])
    start_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = (time.perf_counter() - start_train) / 60

    return pipeline, train_duration

def save_pipeline(pipeline, filepath=None):
    """Sauvegarde le pipeline complet (feature engineering + modèle)."""
    joblib.dump(pipeline, filepath)
    print(f"Pipeline sauvegardé avec succès à : {filepath}")

# def run_pipeline(df, model_name: str = 'gb', best_params = None, lag_months=[1, 2, 3], test_frac=5, save_to=None ):

#     best_params = best_params or getattr(BestParams(), model_name)

#     df_feat, features, le = prepare_data(df, lag_months=lag_months)

#     X_train, y_train, X_test, y_test, train, test = split_data(df_feat, features, test_frac)

#     pipeline, train_duration = train_classifier(X_train, y_train, model_name, best_params)

#     if test_frac != 0.0:
#         print("Evalaution phase")
#         classification_rep, metrics, cm, inference_duration = evaluate_and_plot(pipeline, X_test, y_test, le)
#         return pipeline, le, features, df_feat, test, classification_rep, metrics, cm, best_params, {
#         'train_duration': train_duration,
#         'inference_duration': inference_duration
#     }

#     print("Production phase")
#     save_pipeline(pipeline, save_to)
#     return pipeline, le, features, df_feat, None, None, None, None, best_params, {
#         'train_duration': train_duration,
#         'inference_duration': None
#     }

# lag_months=[1,2,3]
# _, df = get_data(num=300)
# df_feat, features, le = prepare_data(
#                 df, lag_months=lag_months
#             )
# (
#                 X_train,
#                 y_train,
#                 X_test,
#                 y_test,
#                 train,
#                 test,
#             ) = split_data(df_feat, features, 0)
# models = ['gb']
# bestparams = BestParams()
# best_params = getattr(bestparams, 'gb')
# pipeline, train_duration = train_classifier(
#                     X_train,
#                     y_train,
#                     'gb',
#                     best_params,
#                 )
# channel_probs = predict_next_channel_for_client(pipeline, df, 5200000007369, lag_months)
# print(channel_probs)

#######################################################################
############################## Forecasting utils ###################################

