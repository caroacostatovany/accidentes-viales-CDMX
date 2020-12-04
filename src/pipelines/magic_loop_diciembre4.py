
"""
Draft del archivo magic_loop.py

El objetivo es cargar el pickle generado en feature_engineering.py, tirar las columnas irrelevantes,
correr un magic loop e iniciar el resto de las actividades. Hoorray! 

"""

import pandas as pd
import numpy as np 
import pickle
import os
import random
import time
import joblib
from utils import load_df #cambiar por utils.utiles; en mi compu lo hice un poco distinto
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

#from src.pipelines.model_evaluation import plot_roc_auc_curve


COLS_TO_KEEP = ['incidente_c4', 'latitud', 'longitud', 'bool_llamada', 'espacio_del_dia', 
                'label' ]  #delegacion_inicio maybe no

NUM_VARS = ["latitud", "longitud", "bool_llamada", "sin_hr", "cos_hr", "sin_month", "cos_month",
            "sin_day", "cos_day"] 

CAT_VARS = ["incidente_c4", "espacio_del_dia", "delegacion_inicio", "tipo_entrada"] #delegacion_inicio 


#ALGORITHMS = ['tree', 'random_forest'] #sinceramente no funcionaron muy bien; y ya no hay tiempo para entrenarlos
ALGORITHMS = ['logistic', 'tree'] #ya no hay tiempo pa'l Forrest Gump 

PATH = "/Users/enriqueortiz/Documents/PROJECTS/proyecto_ambulancias"
    
#dejé delegación por si es más fácil hacer lo de Aequitas así; lo podemos quitar después 

#lo primero es cargar el pickle

def load_features(path):
    """
    Cargar pickle que se generó durante la transformación
    :param path: Path donde se encuentra el pickle
    :return:
    """
    print("Opening feature engineering pickle from output path")
    output_path = os.path.join(path, "output", "fe_df.pkl")

    # Recuperar el pickle
    incidentes_pkl = load_df(output_path)
    print("Feature Engineering pickle successfully retrieved.")

    return incidentes_pkl


#lo segundo es quedarnos solo con las columnas que vamos a usar

def filter_drop(df):
    """
    Función para elegir variables relevantes y tirar los datos vacíos de latitud y longitud. 
    :param df: dataframe a transformar
    :return df: dataframe con las columnas relevantes y sin datos nulos
    """
    print("Dropping columns and Nan's (don't worry, it'll be ok)")
    #solo por seguridad nos aseguramos que estén ordenadas (aunque ya están)
    df = df.sort_values(by=["año_creacion", "mes_creacion", "dia_creacion",
                            "hora_simple"])
    return df.drop(columns=["año_creacion", "mes_creacion", "dia_creacion"]).dropna()

#el tercer paso sería el famoso magic loop, también conocido como 'Majin Boo' por los fans de Dragon Ball-Z 
#(entre ellos Rhayid Ghani)



#train_test_split
def train_test_split(df, test_size=.70):
    """
    Función para separar en train y test el dataframe.
    Es un poco manual porque son datos temporales -- y no queremos problemas.
    :param df: dataframe a separar en train y test
    :param test_size: fracción entre 0 y 1 que nos permita separar el dataframe. El default es .70
    :return X_train, y_train, X_test, y_test: los 4 fantásticos. 
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("Test Size Error. Pick a test size between 0 and 1.")

    #separar features y labels
    print("Performing train test split")
    X = df.drop(columns=["label"]).copy()
    Y = df.label.copy()
    lim = round(df.shape[0] * test_size)  # 70% de train
    X_train, X_test = X[:lim], X[lim:]
    y_train, y_test = Y[:lim], Y[lim:]
    print("Train test split successfully performed")
    return X_train, y_train, X_test, y_test


#iniciar pipeline de transformación
def transformation_pipeline(X_train, X_test, numerical_attributes, categorical_attributes):
    """
    Crear un pipeline que permita estandarizar el proceso y también recuperar el nombre de las 
    variables.
    :param X_train: features dataframe. 
    :param numerical_attributes: variables numéricas
    :param categorical_attributes: variables categóricas
    :return X_prepared, Y: features (transformadas) y labels
    """
    #esto en realidad no hace nada porque no hay datos nulos, pero facilita lo que viene después
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean"))
        ])

    full_pipeline = ColumnTransformer([
        ("numerical", num_pipeline, numerical_attributes),
        ("categorical", OneHotEncoder(handle_unknown='ignore'), categorical_attributes)
        ])

    print("Features are transformed and ready for magic loop. What about your computer?")
    X_train_reloaded = full_pipeline.fit_transform(X_train)
    X_test_reloaded = full_pipeline.transform(X_test )
    return X_train_reloaded, X_test_reloaded

#ahora sí, ya podemos implementar el Majin Boo
#en el pasado intentamos con el decision tree y el Random Forest, pero no funcionó muy bien
#había muy pocas variables, por lo que decidimos agregar más e intentarlo con Regresión Logística.
#ya no hay tiempo pa' entrenar un Random Forrest Gump.


def save_models(model, path):
    """
    Function to save model from magic loop. 
    """
    joblib.dump(model, path + "/output/model_loop.pkl")





def magic_loop(algorithms, X_train, y_train, X_test, y_test):
    """
    #Función para implementar al famosísimo Majin Boo
    #:param algorithms: list (of algorithms)
    #:X_train, y_train: features and labels of training set, respectively
    #:return: 
    """

    estimators = {'tree': DecisionTreeClassifier(random_state = 1789), \
                   'logistic': LogisticRegression()}
    estimators_params = {'tree': {'max_depth': [3,5,10],
                                    'min_samples_leaf': [1,3,5]},
                          'logistic': {'penalty': ['l2'],
                                       'solver': ['sag', 'saga']}
                       }


    nombres = {'tree': "decision_tree_VF.joblib",
                'logistic': 'logistic_regression_VF.joblib'}

    tscv = TimeSeriesSplit(n_splits=5)
    print("Beginning magic loop. This may take a while.")
    #best_estimators = []
    for algorithm in algorithms:
        print(f"Training model with: {estimators[algorithm]}" )
        estimator = estimators[algorithm]
        grid_params = estimators_params[algorithm]

        gs = GridSearchCV(estimator, grid_params, scoring='precision', cv = tscv,
                          n_jobs = -1)

        start = time.time()
        gs.fit(X_train, y_train)
        print(gs)
        #best_estimators.append(gs)
        joblib.dump(gs, nombres[algorithm])
        print("successfully saved best estimator .")

        if algorithm == 'logistic':
            try: 
                save_models(gs, PATH)
            except:
                pass


        # predicciones con el mejor predictor
        #predicted_labels = gs.predict(X_test)

        # predicciones en score con el mejor predictor
        #predicted_scores = gs.predict_proba(X_test)

        # Ploteamos la curva y guardamos la imagen
        #metrics(estimators[algorithm], y_test,
        #                   predicted_scores, predicted_labels)


        print(f"Total number of seconds: {time.time() - start}")

    #return best_estimators




df = load_features(PATH)
df =  filter_drop(df)
#random.seed(1993)
X_train, y_train, X_test, y_test = train_test_split(df)
X_train, X_test  = transformation_pipeline(X_train, X_test, NUM_VARS, CAT_VARS)
pickle.dump(X_train, open("X_train_diciembre4.pkl", "wb")); pickle.dump(y_train, open("Y_train_diciembre4.pkl", "wb"))
pickle.dump(X_test, open("X_test_diciembre4.pkl", "wb")); pickle.dump(y_test, open("y_test_diciembre4.pkl", "wb"))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
magic_loop(ALGORITHMS, X_train, y_train, X_test, y_test)




