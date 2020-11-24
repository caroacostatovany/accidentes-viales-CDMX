"""
Ingestion module
"""

def ingest_file(file_name):
    """
    Function to retrieve and return the accidents dataset.
    Parameters:
    -----------
    file_name: str
               Path to the file.
    Returns:
    --------
    df: pandas dataframe
    """
    df = pd.read_csv(file_name)
    return df

def drop_cols(df):
    """
    Function to drop unnnecesary columns in the dataset.
    """
    df.drop(columns = ['folio', 'geopoint', 'mes', 'mes_cierre',
                       'hora_cierre', 'año_cierre'], inplace = True)
    return df


def generate_label(df):
    """
    Function to create a new column indicating whether there was
    a false alarm or not.
    Parameters:
    -----------
    df: pandas dataframe

    Returns:
    --------
    df: pandas dataframe
    """
    # transformamos la columna para solo quedarnos con la letra del código
    df["codigo_cierre"] = df["codigo_cierre"].apply(lambda x: x[1])
    df['label'] = np.where(
        (df.codigo_cierre == 'F') | (df.codigo_cierre == 'N'), 1, 0)
    return df

def save_ingestion(df, path):
    """
    Guarda en formato pickle (ver notebook feature_engineering.ipynb) el data frame
    que ya no tiene las columnas que ocuparemos y que incluye el label generado.
    :param df: Dataframe que se utilizará
    :param path:
    :return:
    """
    output_path = os.path.join(path, "output", "ingest_df.pkl")
    # Guardar en el pickle
    pickle.dump(df, open(output_path, "wb"))
