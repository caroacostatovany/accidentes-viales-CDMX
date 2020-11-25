"""
Módulo feature engineering.
Nuestro target es...
Tenemos un problema priorización de recursos? Creo que prevención.
"""
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics

from utils.utils import load_df, save_df

HOURS = 24
MONTHS = 12
DAYS = 7

def load_transformation(path):
    """
    Cargar pickle que se generó durante la transformación
    :param path: Path donde se encuentra el pickle
    :return:
    """
    print("Opening transformation pickle from output path")
    output_path = os.path.join(path, "output", "transformation_df.pkl")

    # Recuperar el pickle
    incidentes_pkl = load_df(output_path)
    print("Pickle successfully retrieved.")

    return incidentes_pkl


def feature_generation(df):
    """
    Crear nuevos features útiles como:
    - Una variable booleana para saber si la llamada es del 911/066 o no.
    - Transformar var categóricas con OneHotEncoder

    :param df: Dataframe del cual se generarán nuevas variables
    :return:
    """

    # Creamos la variable booleana
    print("Creating boolean variable.")
    df["bool_llamada"] = np.where((df.tipo_entrada == "LLAMADA DEL 911") |
                                  (df.tipo_entrada == "LLAMADA DEL 066"), 1, 0)

    print("Transforming discrete variables...")
    # Aplicamos OneHot Encoder para las categóricas
    transformers = [('one_hot', OneHotEncoder(), ['delegacion_inicio', 'incidente_c4',
                                                  'tipo_entrada', 'espacio_del_dia'])]

    col_trans = ColumnTransformer(transformers, remainder="passthrough", n_jobs=-1)

    # Ordenaremos el dataframe temporalmente
    df = df.sort_values(by=["año_creacion", "mes_creacion", "dia_creacion",
                            "hora_simple"])

    X = col_trans.fit_transform(df.drop(columns="label"))
    y = df.label.values.reshape(X.shape[0],)
    print("Successfully transformation of the discrete variables.'")

    print (X.shape)

    return df, X, y


def feature_selection(X, Y):
    """
    Seleccionaremos las variables importantes 
    :param df: Dataframe del que se seleccionarán variables.
    :return:
    """

    # Separación en train y test manualmente para no alterar el tiempo.
    lim = round(data.shape[0] * .70)  # 70% de train
    X_train, X_test = X[:lim], X[lim:]
    y_train, y_test = Y[:lim], Y[lim:]

    # Utilizaremos un Random Forest
    classifier = RandomForestClassifier(oob_score=True, random_state=1993)

    # Definimos los hiperparámetros que queremos probar
    hyper_param_grid = {'n_estimators': [100],
                        'max_depth': [10],
                        'min_samples_split': [2]}

    # Añadimos filtro por tiempo
    tscv = TimeSeriesSplit(n_splits=2)

    # Ocupamos grid search
    gs = GridSearchCV(classifier,
                      hyper_param_grid,
                      scoring='precision',
                      cv=tscv,
                      n_jobs=-1)

    start_time = time.time()
    gs.fit(X_train, y_train)
    print("Time: ", time.time() - start_time)

    return gs


def save_fe(df, path):
    """
    Guardar el dataframe en un pickle
    :param df: Dataframe que ya tiene los features que se ocuparán.
    :param path:
    :return:
    """
    print("Saving feature engineering in pickle format")
    output_path = os.path.join(path, "output", "fe_df.pkl")
    # Guardar en el pickle
    save_df(df,output_path)

    print("Successfully saved fe dataframe as 'fe_df.pkl' in folder 'output'")


def features_removal(df):
    """
    Eliminar más features que no son necesarios
    :param df: Dataframe to adjust
    """
    df = df.drop(['codigo_cierre',
                      'fecha_creacion', 'fecha_cierre',
                      'hora_creacion', 'clas_con_f_alarma',
                      'dia_semana', 'mes_creacion_str',
                      'delegacion_cierre'
                      ], axis=1)

    # df = df[df['incidente_c4'].notna()]

    df['hora_simple'] = df.hora_simple.astype(int)
    df['año_creacion'] = df.año_creacion.astype(int)

    return df


def ciclic_transformation(df):
    """
    Realizar transformaciones cíclicas para hora, mes y día
    :param df:
    """

    print("Cyclic transformation ongoing...")
    df['sin_hr'] = np.sin(2 * np.pi * df.hora_simple / HOURS)
    df['cos_hr'] = np.cos(2 * np.pi * df.hora_simple / HOURS)

    df['sin_month'] = np.sin(2 * np.pi * df.mes_creacion / MONTHS)
    df['cos_month'] = np.cos(2 * np.pi * df.mes_creacion / MONTHS)

    df['sin_day'] = np.sin(2 * np.pi * df.dia_creacion / DAYS)
    df['cos_day'] = np.cos(2 * np.pi * df.dia_creacion / DAYS)

    print("Successfully transformation of the cycle features.'")

    return df


def feature_engineering(path):
    """
    Function to do all the modeling functions
    Parameters:
    -----------
    path: must be the root of the repo
    """
    # Cargamos el picke
    df = load_transformation(path)

    # Eliminamos y convertimos algunos features
    df = features_removal(df)
    df = ciclic_transformation(df)

    # do the feature generation
    df, X, y = feature_generation(df)

    # do the feature selection
    feature_selection(X, y)

    # Guardar el dataframe utilizado
    save_fe(df, path)
    print("Feature Engineering Process Completed")
