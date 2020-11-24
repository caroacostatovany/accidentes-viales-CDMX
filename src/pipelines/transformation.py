"""
Transformation module
"""


def load_ingestion(path):
    """
    Recupera el ingestion file
    """
    output_path = os.path.join(path, "output", "ingest_df.pkl")

    # Recuperar el pickle
    pkl = pickle.load(open(output_path, "rb"))

    return pkl

def date_transformation(col, df):
    """
    Function to prepare and transform date-type columns.
    """
    df[col] = pd.to_datetime(df[col], dayfirst=True)
    return df

def create_date_cols(cols, df):
    """
    Function to transform all date types columns in dataframe
    """
    for col in cols:
        df = date_transformation(col, df)
    return df

def categoric_transformation(col, df):
    """
    Function to transform categorical column
    """
    df[col] = df[col].astype("category")
    return df

def fill_na(df):
    """
    Function to fill null values in a dataframe.
    """
    #aquí podemos ir agregando más cosas cuando descubramos
    #cómo imputar valores faltantes para latitud y longitud
    df.fillna({
        'delegacion_inicio': 'No Disponible',
        'delegacion_cierre': 'No Disponible'
              }, inplace = True)
    return df

def clean_hora_creacion(df):
    """
    Function to transform hours with incorrect format to timedelta format.
    """
    horas_raw = df.hora_creacion.values.tolist()
    horas_clean = [datetime.timedelta(days=float(e)) if e.startswith("0.") else e for e in horas_raw]
    df["hora_creacion"] = horas_clean
    return df


def create_simple_hour(df):
    """
    Function to extract the hour from the column "hora_creacion"
    Parameters:
    -----------
    df: pandas dataframe

    Returns:
    ---------
    df: pandas dataframe with a new column indicating the hour.
    """
    # la función se podria adaptar para devolver minuto o segundo pero no lo considero necesario
    pattern = '\d+'  # encuentra uno o más dígitos
    horas_raw = df.hora_creacion.astype(str).values  # son así: '22:35:04', '22:50:49', '09:40:11'
    n = len(horas_raw)
    horas_clean = [0] * n  # es más rápido reasignar valores que hacer .append()
    for i in range(n):
        hora_raw = horas_raw[i]
        hora_clean = re.match(pattern, hora_raw)[0]  # solo queremos la hora, esto devuelve un objeto
        horas_clean[i] = hora_clean

    df["hora_simple"] = horas_clean
    return df

def save_transformation(df, path):
    """
    Guarda en formato pickle
    el data frame que ya tiene los datos transformados.
    :param df: Dataframe que se utilizará
    :param path:
    :return:
    """
    output_path = os.path.join(path, "output", "transformation_df.pkl")
    # Guardar en el pickle
    pickle.dump(df, open(output_path, "wb"))


def transform(path):
    """
    Function to do all the transformation functions
    Parameters:
    -----------
    path: must be the root of the repo
    """
    df = load_ingestion(path)

    # do all the transformations

    save_transformation(df, path)
