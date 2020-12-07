# Incidentes viales de la Ciudad México 

Noviembre 2020

## **Estructura del proyecto**

+ data
    + **incidentes-viales-c5.csv**
+ docs
    + Data_Profiling.html
+ images
+ notebooks
    + EDA_GEDA_OFICIAL.ipynb
    + feature_engineering.ipynb
+ output
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

**Infraestructura**
  + Se recomienda correrlo en una instancia de 8GB (o más) de RAM.

**Cómo correr el proyecto**
+ Instalar los requirements.txt (en cualquier ambiente virtual)
+ Poner el **archivo incidentes-viales-c5.csv** dentro del folder **data**
+ Ejecutar `python src/proyecto_1.py` desde la ruta principal del repo
+ Si se corren los jupyter notebooks, previamente ejecutar en línea de comandos `export PYTHONPATH=$PWD`


## Proyecto

Los `features` utilizados en este proyecto:
+ incidente_c4
+ bool_llamada (un feature creado con base al tipo de entrada: 1 si es llamada del 066 o del 911, 0 en otro caso)
+ longitud y latitud


#### ¿Qué contiene el proyecto?

Este repositorio contiene una serie de archivos que cargan, transforman y preparan los datos relacionados con incidentes viales de la CDMX para después entrenar una serie de modelos de clasificación, elegir el mejor  y realizar un análisis de sesgo utilizando el framework de Aequitas.

Detallamos algunas de los archivos que se incluyen:

Un archivo html/pdf con el reporte que contiene:
+ La tabla de métricas del mejor modelo (Logistic Regression)
+ La curva ROC
+ La curva de precision y recall
+ Las tablas de métricas obtenidas de la clase `Group` de Aequitas (conteos de frecuencias y absolutas)
+ La visualización de 3 métricas seleccionadas con la salida de `Group`
+ Las tablas de métricas obtenidas de la clase `Bias` de Aequitas (conteos de frecuencias y absolutas)
+ La visualización de 3 métricas seleccionadas con la salida de `Bias` (disparidad)
+ Las tablas de métricas obtenidas de la clase `Fairness` de Aequitas (conteos de frecuencias y absolutas)
+ La visualización de 3 métricas seleccionadas con la salida de `Fairness` (equidad)

+ Una carpeta `docs`:
  + Pdf de *scoping*.
  + *Slide deck* con las cosas más importantes del EDA/GEDA.
  + `html` con tablas de *profiling* y las explicaciones.
+ Una carpeta `notebooks`:
  + `ipynb` con el código para generar *data profilings* y EDA/GEDA
+ Archivo `requirements.txt` 
+ Un *script* `utils.py` que contiene al menos las siguientes funciones:
    + `load_df(path)`: Recibe el *path* en donde se encuentra el `pickle` que se quiere volver a cargar.
    + `save_df(df, path)`: Recibe el `pickle` y el *path* en donde se debe guardar.
+ Un *script* `ingestion.py` que contiene al menos las siguientes funciones:
   + `ingest_file(file_name)`: Recibe un *path* en donde se encuentra el CSV con los datos de reportes de incidentes viales. Regresa un *data frame* con los datos de los incidentes viales.
   + `drop_cols(df)`: Elimina las variables que no ocuparemos.
   + `generate_label()`: Crea en el *data frame* de los datos la variable `label` que es `1` cuando el código de cierre es `(F)` o `(N)`, `0` en otro caso.
   + `save_ingestion(df, path)`: Guarda en formato `pickle` (ver notebook `feature_engineering.ipynb`) el *data frame* que ya no tiene las columnas que ocuparemos y que incluye el `label` generado. El `pickle` se debe llamar `ingest_df.pkl` y se debe guardar en la carpeta `output`.
+ Un *script* `transformation.py` que contiene al menos las siguientes funciones:
  + `load_ingestion(path)`: Recibe el *path* en donde se encuentra el pickle que generamos durante la ingestión.
  + `date_transformation(col, df)`: Recibe la columna que hay que transformar y el *data frame* al que pertenece.
  + `numeric_tranformation(col, df)`: Recibe la columna que hay que transformar y el *data frame* al que pertenece.
  + `categoric_trasformation(col, df)`: Recibe la columna que hay que transformar y el *data frame* al que pertenece.
  + `save_transformation(df, path)`: Guarda en formato `pickle` (ver notebook `feature_engineering.ipynb`) el *data frame* que ya tiene los datos transformados. El `pickle` se debe llamar `transformation_df.pkl` y se debe guardar en la carpeta `output`.
+ Un *script* `feature_engineering.py` que contiene al menos las siguientes funciones:
  + `load_transformation(path)`: Recibe el *path* en donde se encuentra el pickle que generamos durante la transformación.
  + `feature_generation(df)`: Recibe el *data frame* que contiene las variables a partir de las cuales crearemos nuevas variables. Estas nuevas variables se guardarán en este mismo *data frame*.
  + `feature_selection(df)`: Recibe el *data frame*  que contiene las variables de las cuales haremos una selección.
  + `save_fe(df, path)`: Guarda en formato `pickle` (ver notebook `feature_engineering.ipynb`) el *data frame* que ya tiene los *features* que ocuparemos. El `pickle` se debe llamar `fe_df.pkl` y se debe guardar en la carpeta `output`.
+ Un script `modeling.py` que contiene al menos las siguientes funciones:
  + `load_features(path)`: Recibe el *path* en donde se encuentra el `pickle` que generaste durante la selección de *features*.
  + `magic_loop(algorithms)`: Recibe una lista de algoritmos que generarás para predecir si una llamada es falsa.
  + `save_models(model, path)`: Guarda en formato `pickle` el mejor estimador encontrado por tu `magic loop`. El `pickle` se debe llamar `model_loop.pkl` y se debe guardar en la carpeta `output`.
+ Un script `model_evaluation.py` que contiene al menos las siguientes funciones:
  + `load_model(path)`: Recibe el *path* en donde se encuentra el `pickle` que generaste durante el modelado.
  + `metrics(models)`: Genera las siguientes métricas (revisa el notebook `metricas_desempeño.ipynb`):
    + ROC curve
    + Precision
    + Recall
    + Tabla de métricas
  + `save_metrics(df, path)`: Guarda en formato `pickle` el *data frame* de la tabla de métricas de tu modelo seleccionado. El `pickle` se debe llamar `metricas_offline.pkl` y se debe guardar en la carpeta `output`.
+ Un script `bias_fairness.py` que contiene al menos las siguientes funciones:
  + `load_selected_model(path)`: Recibe el *path* en donde se encuentra el `pickle` con el modelo seleccionado en la parte de selección de modelo.
  + `group(df)`: Recibe el *data frame* que tiene los *features* sobre los que queremos medir el sesgo entre los diferentes grupos.   
  + `bias(df)`: Recibe el *data frame* que tiene los *features* sobre los que queremos medir la disparidad
  + `fairness(df)`: Recibe el *data frame* que tiene los *features* sobre los que queremos medir la equidad

+ Un *script* `proyecto_1.py` que manda ejecutar las siguientes funciones:
  + `ingest(path)`: Función en el script `ingestion.py`
  + `transform()`: Función en el script `transformation.py`
  + `feature_engineering()`: Función en el script `feature_engineering.py`
  + `modeling()` Función en el script `modeling.py`
  + `metrics()` Función en el script `model_evaluation.py`
  + `bias_main()`  Función en el script `bias_fairness.py`
