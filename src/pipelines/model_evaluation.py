"""
Modulo para generar las métricas de desempeño
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import pickle

from utils.utils import load_df, save_df, create_images_folder

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve


def load_model(path, file):
    """
    Cargar joblib que se generó durante el modelado
    :param path: Path donde se encuentra el joblib
    :return:
    """
    print("Opening modeling joblib from output path")
    output_path = os.path.join(path, "output", file)

    # Recuperar el pickle
    model_joblib = joblib.load(open(output_path, 'rb'))
    print("Joblib successfully retrieved.")

    return model_joblib


def plot_roc_auc_curve(algorithm, y_test, predicted_scores, fpr, tpr):
    """
    Plotea la curva AUC ROC
    """

    fig = plt.figure(figsize=(10, 5))

    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--', c="red")
    plt.plot(fpr, tpr)
    plt.title("ROC best RF, AUC: {}".format(roc_auc_score(y_test, predicted_scores[:,1])))
    plt.xlabel("fpr")
    plt.ylabel("tpr")

    # Guarda la imagen
    try:
        fig.savefig(r'./images/{}_roc_auc.jpg'.format(algorithm), bbox_inches='tight', dpi=150)
        print("ROC-AUC image successfully saved.")
    except:
        print("ROC-AUC image couldn't be saved.")


def tabla_metricas(fpr, tpr, thresholds, precision, recall, thresholds_2):
    """
    Generar la tabla de métricas (f1score, tpr, fpr, tnr, fnr)
    """

    df_1 = pd.DataFrame({'threshold': thresholds_2,
                         'precision': precision,
                         'recall': recall})
    df_1['f1_score'] = 2 * (df_1.precision * df_1.recall) / (df_1.precision + df_1.recall)

    df_2 = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'threshold': thresholds})
    df_2['tnr'] = 1 - df_2['fpr']
    df_2['fnr'] = 1 - df_2['tpr']

    df = df_1.merge(df_2, on="threshold")

    return df


def save_metrics(df, path):
    """
    Guardar el dataframe en un pickle
    :param df: Metricas en dataframe
    :param path:
    :return:
    """
    print("Saving feature engineering in pickle format")
    output_path = os.path.join(path, "output", "metricas_offline.pkl")
    # Guardar en el pickle
    save_df(df, output_path)

    print("Successfully saved metrics dataframe as 'metricas_offline.pkl' in folder 'output'")


def eficiencia_cobertura():
    """
    Calcular la eficiencia y cobertura en cada punto de corte
    """
    pass


def plot_precision_recall_curve(y_test, predicted_scores):
    """
    Genera la curva de precision recall con el modelo seleccionado incluyendo una línea en k.
    """
    precision, recall, thresholds_2 = precision_recall_curve(y_test,
                                                             predicted_scores[:,1], pos_label=1)

    thresholds_2 = np.append(thresholds_2, 1)


def load_train_test_datasets(path):
    """
    Cargar los datasets para prueba y entrenamiento que se generaron en el modeling
    """
    output_path = os.path.join(path, 'output')

    # Load all X_train, y_train, X_test, y test
    X_train = pickle.load(open(os.path.join(output_path, "X_train.pkl"), 'rb'))
    y_train = pickle.load(open(os.path.join(output_path, "Y_train.pkl"), 'rb'))
    X_test = pickle.load(open(os.path.join(output_path, "X_test.pkl"), 'rb'))
    y_test = pickle.load(open(os.path.join(output_path, "Y_test.pkl"), 'rb'))

    print("train and test datasets successfully retrieved.")

    return X_train, y_train, X_test, y_test


def metrics(path, algorithm='LogisticRegresion'):
    """
    Do all metrics and save them in the proper format:
        ROC Curve (jpg)
        Metrics dataframe (pickle)
        precision_recall_curve (jpg)
    :param path:
    """

    model = load_model(path, "logistic_regression_VF.joblib")
    print("Model Loaded")

    X_train, y_train, X_test, y_test = load_train_test_datasets(path)

    create_images_folder()

    # predicciones con el mejor predictor
    predicted_labels = model.predict(X_test)

    # predicciones en score con el mejor predictor
    predicted_scores = model.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, predicted_scores[:,1], pos_label=1)

    # Ploteamos la curva ROC AUC
    plot_roc_auc_curve(algorithm, y_test, predicted_scores, fpr, tpr)

    precision, recall, thresholds_2 = precision_recall_curve(y_test, predicted_scores[:,1], pos_label=1)
    thresholds_2 = np.append(thresholds_2, 1)

    #Obtenemos la tabla de métricas
    metrics_df = tabla_metricas(fpr, tpr, thresholds, precision, recall, thresholds_2)

    save_metrics(metrics_df, path)

    # Ploteamos la curva de precision y recall en k
    plot_precision_recall_curve(y_test, predicted_scores)
