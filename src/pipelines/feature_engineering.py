"""
Módulo feature engineering.
Nuestro target es...
Tenemos un problema priorización de recursos? Creo que prevención.
"""
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

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
    Crear nuevos features útiles. Transformar var categóricas con OneHotEncoder

    :param df: Dataframe del cual se generarán nuevas variables
    :return:
    """

    transformers = [('one_hot', OneHotEncoder(), ['delegacion_inicio', 'incidente_c4',
                                                  'tipo_entrada', 'espacio_del_dia'])]

    col_trans = ColumnTransformer(transformers, remainder="drop", n_jobs=-1, verbose=True)
    col_trans.fit(df)
    df_input_vars = col_trans.transform(df)

    X = df_input_vars
    y = df.label.values.reshape(df_input_vars.shape[0], )

    return df_input_vars, X, y


def feature_selection(df):
    """
    Seleccionaremos las variables importantes 
    :param df: Dataframe del que se seleccionarán variables.
    :return:
    """
    np.random.seed(1993)

    # Se eliminarán los features con menos del 7%
    variance_threshold = VarianceThreshold(threshold=0.07)
    variance_threshold.fit(df)

    df_vars = variance_threshold.transform(df)

    return df_vars


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
    df = df.drop(['codigo_cierre', 'fecha_creacion', 'fecha_cierre',
                  'hora_creacion', 'clas_con_f_alarma',
                  'año_creacion', 'dia_semana', 'mes_creacion_str',
                  'delegacion_cierre'
                  ], axis=1)

    df = df[df['incidente_c4'].notna()]

    df['hora_simple'] = df.hora_simple.astype(int)

    return df


def ciclic_transformation(df):
    """
    Realizar transformaciones cíclicas
    :param df:
    """

    df['sin_hr'] = np.sin(2 * np.pi * df.hora_simple / HOURS)
    df['cos_hr'] = np.cos(2 * np.pi * df.hora_simple / HOURS)

    df['sin_month'] = np.sin(2 * np.pi * df.mes_creacion / MONTHS)
    df['cos_month'] = np.cos(2 * np.pi * df.mes_creacion / MONTHS)

    df['sin_day'] = np.sin(2 * np.pi * df.dia_creacion / DAYS)
    df['cos_day'] = np.cos(2 * np.pi * df.dia_creacion / DAYS)

    return df

def rfc(X, y):
    """

    """
    classifier = RandomForestClassifier(oob_score=True, random_state=1993)

    # separando en train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # definicion de los hiperparametros que fue el mejor...
    hyper_param_grid = {'n_estimators': [200],
                        'max_depth': [5],
                        'min_samples_split': [2],
                        'criterion': ['gini']}

    # ocupemos grid search!
    gs = GridSearchCV(classifier,
                               hyper_param_grid,
                               scoring = 'precision',
                               cv = 5,
                               n_jobs = -1)
    start_time = time.time()
    gs.fit(X, y)
    print("Tiempo en ejecutar: ", time.time() - start_time)


def feature_engineering(path):
    """
    Function to do all the modeling functions
    Parameters:
    -----------
    path: must be the root of the repo
    """
    df = load_transformation(path)
    df = features_removal(df)
    df = ciclic_transformation(df)

    # do the feature generation
    df, X, y = feature_generation(df)

    # do the feature selection
    df = feature_selection(df)

    rfc(X, y)

    # Guardar el dataframe utilizado
    save_fe(df, path)
