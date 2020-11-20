"""
Módulo feature engineering.
Nuestro target es...
Tenemos un problema priorización de recursos? Creo que prevención.
"""

def load_transformation(path):
    """
    Cargar pickle que se generó durante la transformación
    :param path: Path donde se encuentra el pickle
    :return:
    """


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

def remove_variables_non_useful(incidentes_df):
    """

    :param incidentes_df:
    :return:
    """
    incidentes_df = incidentes_df.drop(['geopoint', 'folio',
                                        'mes_cierre', 'clas_con_f_alarma',
                                        'fecha_cierre', 'año_cierre',
                                        'mes_cierre', 'hora_cierre',
                                        'delegacion_cierre', 'mes'], axis=1)

    return incidentes_df

def separar_mes_anio_creacion(incidentes_df):
    incidentes_df['mes_creacion'] = incidentes_df.fecha_creacion.dt.strftime('%m')
    incidentes_df["anio_creacion"] = incidentes_df.fecha_creacion.dt.strftime('%Y')

    return incidentes_df