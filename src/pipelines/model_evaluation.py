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
from sklearn.metrics import precision_score, recall_score

ambulancias_disponibles_al_dia = 20.0
num_llamadas_al_dia = 554.0

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

def precision_at_k(y_true, y_scores, k):
    """
    Obtain precision at k
    """
    threshold = np.sort(y_scores)[::-1][int(k*(len(y_scores)-1))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])

    return precision_score(y_true, y_pred)

def recall_at_k(y_true, y_scores, k):
    """
    Obtain recall at k
    """
    threshold = np.sort(y_scores)[::-1][int(k*(len(y_scores)-1))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])

    return recall_score(y_true, y_pred)

def plot_precision_recall_curve(y_true, y_scores, metrics_report):
    """
    Genera la curva de precision recall con el modelo seleccionado incluyendo una línea en k.
    """
    k_values = list(np.arange(0.1, 1.1, 0.1))
    pr_k = pd.DataFrame()

    for k in k_values:
        d = dict()
        d['k'] = k
        ## get_top_k es una función que ordena los scores de
        ## mayor a menor y toma los k% primeros
        #top_k = get_top_k(y_scores, k)
        #top_k = y_scores
        # print(precision_at_k(y_true, y_scores, k))
        d['precision'] = precision_at_k(y_true, y_scores, k)#(top_k)
        d['recall'] = recall_at_k(y_true, y_scores, k)#(top_k, predictions)

        pr_k = pr_k.append(d, ignore_index=True)

    # para la gráfica
    fig, ax1 = plt.subplots()
    ax1.plot(pr_k['k'], pr_k['precision'], label='precision')
    ax1.plot(pr_k['k'], pr_k['recall'], label='recall')
    plt.legend()
    plt.savefig(r'./images/precision_recall_1.jpg', dpi=300)


    fig, ax1 = plt.subplots()
    ax1.plot(metrics_report.threshold, metrics_report.precision, label="Precision")
    ax1.plot(metrics_report.threshold, metrics_report.recall, label="Recall")
    ax1.plot(metrics_report.threshold, metrics_report.tnr, label="True negative rate")
    plt.axvline(x=k, color="red", linestyle="--", label="k")
    plt.legend(loc="best")
    plt.xlabel("k")
    plt.ylabel("metric value")
    plt.savefig(r'./images/precision_recall_2.jpg', dpi=300)
    print("Successfully saved precision_recall@k plot in 'images'.")

    return pr_k


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
    plot_precision_recall_curve(predicted_labels, y_test, metrics_df)
