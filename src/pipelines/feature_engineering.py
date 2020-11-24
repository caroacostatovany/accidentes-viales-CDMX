"""
Módulo feature engineering.
Nuestro target es...
Tenemos un problema priorización de recursos? Creo que prevención.
"""
from utils import load_df, save_df

def load_transformation(path):
    """
    Cargar pickle que se generó durante la transformación
    :param path: Path donde se encuentra el pickle
    :return:
    """
    output_path = os.path.join(path, "transformation_df.pkl")

    # Recuperar el pickle
    incidentes_pkl = load_df(output_path)

    return incidentes_pkl


def feature_generation(df):
    """
    Crear nuevos features útiles.

    :param df: Dataframe del cual se generarán nuevas variables
    :return:
    """

    return df


def feature_selection(df):
    """
    Seleccionaremos las variables importantes
    :param df: Dataframe del que se seleccionarán variables.
    :return:
    """

def save_fe(df, path):
    """
    Guardar el dataframe en un pickle
    :param df: Dataframe que ya tiene los features que se ocuparán.
    :param path:
    :return:
    """
    output_path = os.path.join(path, "output", "fe_df.pkl")
    # Guardar en el pickle
    save_df(df,output_path)

