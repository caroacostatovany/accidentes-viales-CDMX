"""
Ingestion module
"""
import os
import pandas as pd
import numpy as np
import pickle

from utils.utils import MAPPING_MESES

def ingest_file(path):
    """
    Function to retrieve and return the accidents dataset.
    Parameters:
    -----------
    path: str
               Path to the file.
    Returns:
    --------
    df: pandas dataframe
    """
    print("Reading data...")
    # os.chdir(path)
    df = pd.read_csv(path)
    return df


def create_output_folder():
    """
    Function to create a folder if it doesn't exist. 
    """
    if not os.path.exists("output"):
        os.mkdir("output")
        print("Creating output folder (It didn't exist!)")
    

def drop_cols(df):
    """
    Function to drop unnnecesary columns in the dataset.
    """
    df.drop(columns = ['folio', 'geopoint', 'mes', 'mes_cierre',
                       'hora_cierre', 'año_cierre'], inplace = True)
    print("Dropping columns...")
    print("Columns dropped.")
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
    print("Generating label.")
    df["codigo_cierre"] = df["codigo_cierre"].apply(lambda x: x[1])
    df['label'] = np.where(
        (df.codigo_cierre == 'F') | (df.codigo_cierre == 'N'), 1, 0)
    print("Label generated successfully.")
    return df

def save_ingestion(df, path):
    """
    Guarda en formato pickle (ver notebook feature_engineering.ipynb) el data frame
    que ya no tiene las columnas que ocuparemos y que incluye el label generado.
    :param df: Dataframe que se utilizará
    :param path:
    :return:
    """
    create_output_folder()
    #output_path = os.path.join(path, "output", "ingest_df.pkl")
    # Guardar en el pickle
    print("Saving pickle in output folder")
    os.chdir(os.path.join(path, "output"))
    pickle.dump(df, open("ingest_df.pkl", "wb"))
    print("Pickle saved in output folder.")

def add_date_columns(df):
    """
    Esta función es muy importante puesto que nos ayudará a crear el mes, día y año de creación
    del registro. De esta manera podemos prescindir de las fechas de cierre, que no tendríamos en tiempo
    real en un modelo.
    Parameters:
    -----------
    df: pandas dataframe

    Returns:
    ---------
    df: pandas dataframe with 4 new columns
    """

    df["año_creacion"] = df.fecha_creacion.dt.year
    df["mes_creacion"] = df.fecha_creacion.dt.month
    df["dia_creacion"] = df.fecha_creacion.dt.day
    df["mes_creacion_str"] = df.mes_creacion.map(MAPPING_MESES)
    df["año_creacion"] = df["año_creacion"].astype(str)
    return df


def create_time_blocks(df):
    """
    Function to group the hour of the day into 3-hour blocks.
    Parameters:
    -----------
    df: pandas dataframe

    Returns:
    ---------
    df: pandas dataframe with a new column indicating the time-block.
    """
    horas_int = set(df.hora_simple.astype(int).values)  # estaba como categórico
    f = lambda x: 12 if x == 0 else x
    mapping_hours = {}
    for hora in horas_int:
        grupo = (hora // 3) * 3
        if grupo < 12:
            nombre_grupo = str(f(grupo)) + "-" + str(grupo + 2) + " a.m."
        else:
            hora_tarde = grupo % 12
            nombre_grupo = str(f(hora_tarde)) + "-" + str(hora_tarde + 2) + " p.m."
        mapping_hours[hora] = nombre_grupo

    df["espacio_del_dia"] = df["hora_simple"].astype(int).map(mapping_hours)
    return df


def ingest(path, file_name):
    """
    Function to do all ingestion functions
    Parameters:
    -----------
    path: must be the root of the repo
    """
    data_path = os.path.join(path, 'data', file_name)

    df = ingest_file(data_path)
    df = generate_label(df)
    df = drop_cols(df)

    save_ingestion(df, path)
    print("Ingestion Process Completed")
    print("--"*30)



# ingest("/Users/enriqueortiz/Documents/PROJECTS/proyecto_ambulancias/data/", "incidentes-viales-c5.csv")







