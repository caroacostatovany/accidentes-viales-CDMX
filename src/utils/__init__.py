"""
Funciones útiles generales.
"""
import numpy as np

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


def generate_label(incidentes_viales_df):
    """
    Cambiar el codigo de cierre a
    :param incidentes_viales_df: dataframe
    :return:
    """
    incidentes_viales_df.codigo_cierre.mask(incidentes_viales_df.codigo_cierre ==
                                            r"(A) La unidad de atención a emergencias fue despachada, "
                                            "llegó al lugar de los hechos y confirmó la emergencia reportada",
                                            'A', inplace=True)
    incidentes_viales_df.codigo_cierre.mask(incidentes_viales_df.codigo_cierre ==
                                            r'(N) La unidad de atención a emergencias fue despachada, '
                                            'llegó al lugar de los hechos, pero en el sitio del evento '
                                            'nadie solicitó el apoyo de la unidad',
                                            'N', inplace=True)
    incidentes_viales_df.codigo_cierre.mask(incidentes_viales_df.codigo_cierre ==
                                            r'(D) El incidente reportado se registró en dos o más '
                                            'ocasiones procediendo a mantener un único reporte (afirmativo,'
                                            ' informativo, negativo o falso) como el identificador para el '
                                            'incidente',
                                            'D', inplace=True)
    incidentes_viales_df.codigo_cierre.mask(incidentes_viales_df.codigo_cierre ==
                                            r'(F) El operador/a o despachador/a identifican, antes de dar '
                                            'respuesta a la emergencia, que ésta es falsa. O al ser '
                                            'despachada una unidad de atención a emergencias en el lugar '
                                            'de los hechos se percatan que el incidente no corresponde al '
                                            'reportado inicialmente',
                                            'F', inplace=True)
    incidentes_viales_df.codigo_cierre.mask(incidentes_viales_df.codigo_cierre ==
                                            r'(I) El incidente reportado es afirmativo y se añade '
                                            'información adicional al evento',
                                            'I', inplace=True)

    incidentes_viales_df['label'] = np.where(
        (incidentes_viales_df.codigo_cierre == 'F') | (incidentes_viales_df.codigo_cierre == 'N'), 1, 0)

    return incidentes_viales_df

