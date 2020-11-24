"""
Funciones útiles generales.
"""
import os
import numpy as np
import pickle
import pandas as pd

CAT_COLS = ["dia_semana", "codigo_cierre", "año_cierre", "mes_cierre", "mes", "delegacion_inicio",
            "incidente_c4", "clas_con_f_alarma", "tipo_entrada", "delegacion_cierre", "hora_creacion",
           "hora_cierre"]

DATE_COLS = ["fecha_creacion", "fecha_cierre"]

NUM_COLS = ["latitud", "longitud"]

MAPPING_MESES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo",
                 6: "Junio", 7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
                 11: "Noviembre", 12: "Diciembre"}

def number_formatter(number, pos=None):
    """
    Convert a number into a human readable format.
    :param number:
    :param pos:
    :return: string converted
    """
    magnitude = 0
    while abs(number) >= 1000:
        magnitude += 1
        number /= 1000.0
    return '%.1f%s' % (number, ['', 'K', 'M', 'B', 'T', 'Q'][magnitude])


def numeric_profiling(df_o, col):
    """

    :param df_o:
    :param col: column to analize
    :return: dictionary
    """
    profiling = {}

    # eliminate missing values
    df = df_o.copy()
    df = df[df[col].notna()]
    df[col] = df[col].astype(float)

    profiling.update({'max': df[col].max(),
                     'min': df[col].min(),
                     'mean': df[col].mean(),
                     'stdv': df[col].std(),
                     '25%': df[col].quantile(.25),
                     'median': df[col].median(),
                     '75%': df[col].quantile(.75),
                     'kurtosis': df[col].kurt(),
                     'skewness': df[col].skew(),
                     'uniques': df[col].nunique()}
                     # 'prop_missings': df[col].isna().sum()/df.shape[0]*100,
                     # 'top1_repeated': get_repeated_values(df, col, 1),
                     # 'top2_repeated': get_repeated_values(df, col, 2),
                     # 'top3_repeated': get_repeated_values(df, col, 3)
                     )

    return profiling

def load_df(path):
    """
    Recibe el path en donde se encuentra el pickle que se quiere volver a cargar.
    """
    # Recuperar el pickle
    pkl = pickle.load(open(path, "rb"))

    return pkl

def save_df(df, path):
    """
    Guardar el dataframe en un pickle
    :param df: Dataframe que ya se guardará
    :param path:
    :return:
    """
    # Guardar en el pickle
    pickle.dump(df, open(path, "wb"))


def numeric_profiling(df, col):
    """
    Profiling for numeric columns.

    :param: column to analyze
    :return: dictionary
    """
    profiling = {}

    profiling.update({'max': df[col].max(),
                      'min': df[col].min(),
                      'mean': df[col].mean(),
                      'stdv': df[col].std(),
                      '25%': df[col].quantile(.25),
                      'median': df[col].median(),
                      '75%': df[col].quantile(.75),
                      'kurtosis': df[col].kurt(),
                      'skewness': df[col].skew(),
                      'uniques': df[col].nunique(),
                      'prop_missings': df[col].isna().sum() / df.shape[0] * 100,
                      'top1_repeated': get_repeated_values(df, col, 1),
                      'top2_repeated': get_repeated_values(df, col, 2), })

    return profiling


def get_repeated_values(df, col, top):
    """
    Function to obtain top 3 values
    """
    top_5 = df.groupby([col])[col] \
        .count() \
        .sort_values(ascending=False) \
        .head(3)
    indexes_top_5 = top_5.index

    if ((top == 1) and (len(indexes_top_5) > 0)):
        return indexes_top_5[0]
    elif ((top == 2) and (len(indexes_top_5) > 1)):
        return indexes_top_5[1]
    elif ((top == 3) and (len(indexes_top_5) > 2)):
        return indexes_top_5[2]
    else:
        return 'undefined'

