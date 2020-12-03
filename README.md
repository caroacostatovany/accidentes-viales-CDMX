Noviembre 2020

## **Estructura del proyecto**


+ data
    + **incidentes-viales-c5.csv**
+ docs
    + Data_Profiling.html
+ images
    + 
+ notebooks
    + EDA_GEDA_OFICIAL.ipynb
    + feature_engineering.ipynb
+ output
    +
    +
    +
+ src
    + pipelines
        + feature_engineering.py
        + ingestion.py
        + magic_loop.py
        + transformation.py
    + utils
        + utils.py
    + proyect_1.py
    
+ requirements.txt

## **Equipo**

|Nombre|Usuario Github| ID |
|------|--------------|----|
| Leonardo Ceja Pérez | lecepe00 | 000197818 |
| Enrique Ortiz Casillas | EnriqueOrtiz27 | 000150644 |
| Carolina Acosta Tovany | caroacostatovany | 000197627 |

## **Para correr el proyecto**

+ Instalar los requirements.txt 
+ Poner el **archivo incidentes-viales-c5.csv** dentro del folder **data**
+ Ejecutar `python src/proyecto_1.py`
+ Si se corren los jupyter notebooks, previamente ejecutar en línea de comandos `export PYTHONPATH=$PWD`


## Proyecto 2

Los `features` utilizados en este proyecto:
+ incidente_c4
+ bool_llamada (un feature creado con base al tipo de entrada: 1 si es llamada del 066 o del 911, 0 en otro caso)
+ longitud y latitud


**Modelado**
1. Genera un `magic loop` con los siguientes 2 algoritmos cada uno modificando al menos 2 hiperparámetros con al menos 3 configuraciones diferentes y utilizando un CV de 5 a través de un `GridSearchCV`, -toma en cuenta que este CV debe ser un *temporal split*. El criterio de selección de mejor modelo será la eficiencia del modelo:   
  + DecisionTree
  + RandomForest

**Evaluación de modelo**
2. Generar la curva ROC, la tabla de métricas del modelo y la eficiencia y cobertura en cada punto de corte.   
3. Generar la curva de precision recall con el modelo seleccionado incluyendo una línea en tu `k`.
+ ¿Cuánto tienes de cobertura@k? Interpreta
+ ¿Cuánto tienes de eficiencia@k? Interpreta
4. ¿Cuál es la cobertura máxima que se puede tener con la `k` que tienes?
+ Tomando en cuenta la respuesta a esta pregunta, tu modelo ¿es bueno? ¿es malo?

**Sesgo e inequidad**
5. Utilizando el *feature* `delegación` como atributo protegido:
a. Selecciona 3 métricas **adecuadas** de acuerdo a los objetivos y acciones. Justifica.
b. Define cuál es tu grupo de referencia. Justifica.
c. Cuantifica el sesgo e inquedidad sobre los grupos de este atributo.


#### ¿Qué se entrega?

A entregarse máximo el **6 de diciembre de 2020 23:59:59 CST**. Del mejor de tus modelos seleccionado.

Un archivo html/pdf con el reporte que contiene:
+ La tabla de métricas de tu mejor modelo
+ La curva ROC
+ La curva de precision y recall
+ Las tablas de métricas obtenidas de la clase `Group` de Aequitas (conteos de frecuencias y absolutas)
+ La visualización de tus 3 métricas seleccionadas con la salida de `Group`
+ Las tablas de métricas obtenidas de la clase `Bias` de Aequitas (conteos de frecuencias y absolutas)
+ La visualización de tus 3 métricas seleccionadas con la salida de `Bias` (disparidad)
+ Las tablas de métricas obtenidas de la clase `Fairness` de Aequitas (conteos de frecuencias y absolutas)
+ La visualización de tus 3 métricas seleccionadas con la salida de `Fairness` (equidad)

Actualización a tu repositorio con el siguiente código.

+ Un script `modeling.py` que tenga las siguientes funciones:
  + `load_features(path)`: Recibe el *path* en donde se encuentra el `pickle` que generaste durante la selección de *features*.
  + `magic_loop(algorithms)`: Recibe una lista de algoritmos que generarás para predecir si una llamada es falsa.
  + `save_models(model, path)`: Guarda en formato `pickle` el mejor estimador encontrado por tu `magic loop`. El `pickle` se debe llamar `model_loop.pkl` y se debe guardar en la carpeta `output`.
+ Un script `model_evaluation.py` que tenga las siguientes funciones:
  + `load_model(path)`: Recibe el *path* en donde se encuentra el `pickle` que generaste durante el modelado.
  + `metrics(models)`: Genera las siguientes métricas (revisa el notebook `metricas_desempeño.ipynb`):
    + ROC curve
    + Precision
    + Recall
    + Tabla de métricas
  + `save_metrics(df, path)`: Guarda en formato `pickle` el *data frame* de la tabla de métricas de tu modelo seleccionado. El `pickle` se debe llamar `metricas_offline.pkl` y se debe guardar en la carpeta `output`.
+ Un script `bias_fairness.py` que tenga las siguientes funciones (revisa el [notebook de aequitas](https://github.com/dssg/aequitas/blob/master/docs/source/examples/compas_demo.ipynb )):
  + `load_selected_model(path)`: Recibe el *path* en donde se encuentra el `pickle` con el modelo seleccionado en la parte de selección de modelo.
  + `group(df)`: Recibe el *data frame* que tiene los *features* sobre los que queremos medir el sesgo entre los diferentes grupos.   
  + `bias(df)`: Recibe el *data frame* que tiene los *features* sobre los que queremos medir la disparidad
  + `fairness(df)`: Recibe el *data frame* que tiene los *features* sobre los que queremos medir la equidad


Archivo README.md
+ Pequeña explicación de qué es el contenido del repositorio.
+ Qué infraestructura es necesaria para correr su código
+ Qué infraestructura se necesita para poder correr su código? (características de la máquina)
+ Cómo correr su código


**Scripts**

![](./images/proyecto_2_scripts.png)

**Estructura de carpetas**

![](./images/proyecto_2_file_structure.png)

**Equipos**

![](./images/equipos.png)

**Precision@k**

Autor: Rayid Ghani
```
def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])

    return metrics.precision_score(y_true, y_pred)
```

**Recall@k**

Autor: Rayid Ghani
```
def recall_at_k(y_true, y_scores, k):
   threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
   y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])

   return metrics.recall_score(y_true, y_pred)
```

**Precision-recall@k curve**

Autor: Kasun Amarasinghe
```
def pr_k_curve(y_true, y_scores, save_target)
  k_values = list(np.arange(0.1, 1.1, 0.1))
  pr_k = pd.DataFrame()

    for k in k_values:
        d = dict()
        d['k'] = k
        ## get_top_k es una función que ordena los scores de
        ## mayor a menor y toma los k% primeros
        top_k = get_top_k(y_scores, k)
        d['precision'] = precision_at_k(top_k)
        d['recall'] = recall_at_k(top_k, predictions)

        pr_k = pr_k.append(d, ignore_index=True)

    # para la gráfica
    fig, ax1 = plt.subplots()
    ax1.plot(pr_k['k'], pr_k['precision'], label='precision')
    ax1.plot(pr_k['k'], pr_k['recall'], label='recall')
    plt.legend()

    if save_target is not None:
        plt.savefig(save_target, dpi=300)

    return pr_k
```
