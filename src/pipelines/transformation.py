"""
Transformation module
"""
import os
import pickle
import pandas as pd
import datetime
import re

from utils.utils import DATE_COLS

CAT_COLS = ["dia_semana", "codigo_cierre", "delegacion_inicio",
            "incidente_c4", "clas_con_f_alarma", "tipo_entrada", "delegacion_cierre", "hora_creacion",
           ]


def load_ingestion(path):
    """
    Recupera el ingestion file
    """
    output_path = os.path.join(path, "output", "ingest_df.pkl")

    # Recuperar el pickle
    print("Opening pickle from output path")
    pkl = pickle.load(open(output_path, "rb"))
    print("Pickle successfully retrieved.")

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

def fillna(df):
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
    horas_int = set(df.hora_simple.astype(int).values) #estaba como categórico
    f = lambda x: 12 if x == 0 else x
    mapping_hours = {}
    for hora in horas_int:
        grupo = (hora // 3) * 3
        if grupo < 12: 
            nombre_grupo = str(f(grupo)) + "-" + str(grupo + 3) + " a.m."
        else:
            hora_tarde = grupo % 12
            nombre_grupo = str(f(hora_tarde)) + "-" + str(hora_tarde + 3) + " p.m."
        mapping_hours[hora] = nombre_grupo
    
    df["espacio_del_dia"] = df["hora_simple"].astype(int).map(mapping_hours)
    return df
    



def categoric_transformation(col,df):
    df[col] = df[col].astype("category")
    return df 

def create_categorical(cols, df):
    """
    Function to transform and prepare the categorical features in the dataset.
    """
    #transform to appropriate data type
    for col in cols: 
        df = categoric_transformation(col, df)
     
    return df



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
    mapping_meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo",
                       6: "Junio", 7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
                       11: "Noviembre", 12: "Diciembre"}
    
    
    df["año_creacion"] = df.fecha_creacion.dt.year
    df["mes_creacion"] = df.fecha_creacion.dt.month
    df["dia_creacion"] = df.fecha_creacion.dt.day
    df["mes_creacion_str"] = df.mes_creacion.map(mapping_meses)
    df["año_creacion"] = df["año_creacion"].astype(str)
    return df 




def appply_all_transformations(df):
    """
    Function to execute the above transformations. 
    :param df: dataframe del proceso
    :return df: dataframe transformado
    """
    df = fillna(df)
    df = clean_hora_creacion(df)
    df = create_categorical(CAT_COLS, df)
    df = create_date_cols(DATE_COLS, df)
    df = add_date_columns(df)
    df = create_simple_hour(df)
    df = create_time_blocks(df)
    return df 


def save_transformation(df, path):
    """
    Guarda en formato pickle
    el data frame que ya tiene los datos transformados.
    :param df: Dataframe que se utilizará
    :param path:
    :return:
    """
    print("Saving transformation in pickle format.")
    output_path = os.path.join(path, "output", "transformation_df.pkl")

    # Guardar en el pickle
    pickle.dump(df, open(output_path, "wb"))
    print("Successfully saved transformed dataframe as 'transformation_df.pkl' in folder 'output'.")


def transform(path):
    """
    Function to do all the transformation functions
    Parameters:
    -----------
    path: must be the root of the repo
    """
    df = load_ingestion(path)

    # do all the transformations
    print("Performing transformations on the dataframe. Please wait.")
    df = appply_all_transformations(df)
    print("Successfully transformed the dataframe")

    # Save the transformations
    save_transformation(df, path)

# path = "/Users/enriqueortiz/Documents/PROJECTS/proyecto_ambulancias/data"
# transform(path)
