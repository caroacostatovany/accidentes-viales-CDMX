"""
Modulo para generar las métricas de desempeño
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve

def plot_roc_auc_curve(algorithm, y_test, predicted_scores, predicted_labels):
    """
    Plotea la curva AUC ROC
    """

    fpr, tpr, thresholds = roc_curve(y_test, predicted_scores[:, 1], pos_label=1)

    fig = plt.figure(figsize=(10, 5))

    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--', c="red")
    plt.plot(fpr, tpr)
    plt.title("ROC best RF, AUC: {}".format(roc_auc_score(y_test, predicted_labels)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")

    # Guarda la imagen
    fig.savefig('../images/{}_roc_auc.jpg'.format(algorithm), bbox_inches='tight', dpi=150)

def tabla_metricas(fpr, tpr, thresholds, precision, recall, thresholds_2):
    """

    """
    df_1 = pd.DataFrame({'threshold': thresholds_2, 'precision': precision,
                         'recall': recall})
    df_1['f1_score'] = 2 * (df_1.precision * df_1.recall) / (df_1.precision + df_1.recall)

    df_2 = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'threshold': thresholds})
    df_2['tnr'] = 1 - df_2['fpr']
    df_2['fnr'] = 1 - df_2['tpr']

    df = df_1.merge(df_2, on="threshold")

    return df


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






