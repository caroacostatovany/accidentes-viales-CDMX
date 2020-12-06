

import seaborn as sns
import os 
import pickle
import joblib
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
from sklearn.metrics import confusion_matrix

#funciones de apoyo 
def load_df(path):
    """
    Recibe el path en donde se encuentra el pickle que se quiere volver a cargar.
    """
    # Recuperar el pickle
    pkl = pickle.load(open(path, "rb"))

    return pkl



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



def load_selected_model(path):
    model = joblib.load(path) 
    #hacerlo con pickle?
    return model



def generate_aequitas_df(path, model):
    """
    Function to generate an Aequitas-compliant dataframe
    :param path: root
    :model: a model to predict new labels
    :return: aeq_df: Aequitas-compliant dataframe
    """
    #recover original dataframe
    path = os.path.join(path, 'output', 'transformation_df.pkl')
    df = load_df(path)
    df = filter_drop(df)
    #separar en train y test
    X_train, y_train, X_test, y_test = train_test_split(df)
    #realizar predicciones y comparar
    predicted_scores = model.predict_proba(X_test)
    predicted_probs = pd.DataFrame(predicted_scores, columns=["probability_0", "probability_1"])
    punto_corte = .2903
    new_labels = [0 if score < punto_corte else 1 for score in predicted_probs.probability_1]

    delegaciones = df.iloc[len(new_labels):,:].delegacion_inicio
    aeq_df = pd.DataFrame({"score": new_labels, 
                       "label_value": y_test.values,
                       "delegacion": delegaciones}
                    ).reset_index().iloc[:,1:]
    return aeq_df


def group(df):
    """
    Function to print absolute and relative metrics by group.
    :param df: aequitas-compliant dataframe.
    :return: None
    """
    g = Group()
    print("Módulo de grupo")
    print("-"*30)
    xtab, atts = g.get_crosstabs(df, attr_cols=["delegacion"])
    absolute_metrics = g.list_absolute_metrics(xtab)
    print(f"El grupo a analizar es:{atts}")
    print("Conteo de frecuencias por grupos:")
    print(xtab[[col for col in xtab.columns if col not in absolute_metrics]])
    print()
    print("Conteo de absolutos por grupos:")
    print(xtab[['attribute_name', 'attribute_value']+[col for col in xtab.columns if col in absolute_metrics]].round(2))



def bias(df):
    """
    Function to print bias metrics. 
    :param df: Aequitas-compliant dataframe.  
    """
    
    print("Módulo de Sesgo")
    print("-"*30)
    bias_ = Bias()
    g = Group()
    xtab, atts = g.get_crosstabs(df, attr_cols=["delegacion"])
    absolute_metrics = g.list_absolute_metrics(xtab)
    bdf = bias_.get_disparity_predefined_groups(xtab, original_df = df, 
                                ref_groups_dict = {'delegacion': 'Iztapalapa'}, 
                                alpha=0.05)
    print("Disparities:")
    print(bdf[['attribute_name', 'attribute_value'] + bias_.list_disparities(bdf)].round(2))
    print("Minority Analysis:")
    min_bdf = bias_.get_disparity_min_metric(xtab, original_df=df)
    print(min_bdf[['attribute_name', 'attribute_value'] +  bias_.list_disparities(min_bdf)].round(2))
    print("Majority Analysis:")
    majority_bdf = bias_.get_disparity_major_group(xtab, original_df=df)
    print(majority_bdf[['attribute_name', 'attribute_value'] +  bias_.list_disparities(majority_bdf)].round(2))



def fairness(df):
    print("Módulo de Equidad")
    print("-"*30)
    f = Fairness()
    bias_ = Bias()
    g = Group()
    xtab, atts = g.get_crosstabs(df, attr_cols=["delegacion"])
    absolute_metrics = g.list_absolute_metrics(xtab)
    bdf = bias_.get_disparity_predefined_groups(xtab, original_df = df, 
                                ref_groups_dict = {'delegacion': 'Iztapalapa'}, 
                                alpha=0.05)
    fdf = f.get_group_value_fairness(bdf)
    parity_determinations = f.list_parities(fdf)
    print("Imprimiendo tabla de métricas (conteos en frecuencias):")
    print(fdf[['attribute_name', 'attribute_value'] + absolute_metrics + bias_.list_disparities(fdf) + parity_determinations].round(2))
    print("Impriendo métricas generales")
    gof = f.get_overall_fairness(fdf)
    print(gof)
    print("Aequitas analysis completed.")
    print("-"*30)



#pruebas
path = 'hola'
modelo = load_selected_model(path)
df = generate_aequitas_df(path, modelo)
group(df)
bias(df)
fairness(df)




