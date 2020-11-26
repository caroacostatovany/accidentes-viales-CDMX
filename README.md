![](./images/itam_logo.png)

### Proyecto 1

Contamos con los datos de accidentes viales en la CDMX reportados por C5.

De la página de [datos.cdmx.gob.mx](https://datos.cdmx.gob.mx)

En este conjunto se reporta: folio, fecha de creación del reporte, hora de creación del reporte, día de la semana de creación del reporte, fecha de cierre de reporte, hora de cierre de reporte, motivo del incidente dependiendo del tipo de emergencia, alcaldía donde sucedió el incidente, latitud y longitud del incidente, código de cierre del incidente reportado, clasificación del incidente, origen del incidente por tipo, alcaldía en que se dio resolución al incidente o emergencia.

Los registros que se reciben en el C5 se clasifican internamente por medio de un código de cierre:

A = “Afirmativo”: Una unidad de atención a emergencias fue despachada, llegó al lugar de los hechos y confirmó la emergencia reportada

N = “Negativo”: Una unidad de atención a emergencias fue despachada, llegó al lugar de los hechos, pero en el sitio del evento nadie confirmo la emergencia ni fue solicitado el apoyo de la unidad

I = “Informativo”: Corresponde a solicitudes de información

F = “Falso”: El incidente reportado inicialmente fue considerado como falso en el lugar de los hechos.

D = “Duplicados”: El incidente reportado se registró en dos o más ocasiones procediendo a mantener un solo reporte como el original. Para el uso e interpretación correctos de la información, debe considerarse:

    1) Para contabilizar aquellos incidentes “reales”, es decir, aquellos confirmados por las autoridades en el lugar de los hechos, se debe trabajar con los códigos de cierre “Afirmativo”.
    2) Para contabilizar los registros recibidos en la institución, se debe trabajar con todos los códigos de cierre.

¿Qué es el C5?

l Centro de Comando, Control, Cómputo, Comunicaciones y Contacto Ciudadano de la CDMX (C5), es la dependencia del Gobierno de la Ciudad de México encargada de captar información integral para la toma de decisiones en materia de seguridad pública, urgencias médicas, medio ambiente, protección civil, movilidad y servicios a la comunidad en la capital del país a través del video monitoreo, de la captación de llamadas telefónicas y de aplicaciones informáticas de inteligencia, enfocadas a mejorar la calidad de vida de las y los capitalinos. Cada incidente registrado genera un folio único identificador, excepto las llamadas no procedentes o “falsas” (bromas, niños jugando, etc.) que se reciben por distintos medios.  Mientras un folio está abierto; es decir, mientras el registro está siendo atendido, éste no se carga en la base de datos interna, una vez que el folio ha sido cerrado éste se refleja en el sistema y es posible visibilizarlo internamente con un día de vencimiento, pues se realiza una recarga diaria (por esta razón las fechas de inicio y cierre de folio no necesariamente coinciden).

El diccionario de datos así como los datos se encuentran en la siguiente liga de [dropbox](https://www.dropbox.com/sh/sj3q1y6gilv6yfv/AABBmm9fGuWzWc6Ueh7xHiBba?dl=0).

***

Queremos predecir si una llamada al C5 para reportar un inicidente vial es **Falsa** o no.

Información relevante:

+ Solo tenemos **20 ambulancias** para enviar en caso de un incidente.

**Acciones**

+ En caso de que la llamada sea verdadera, se envía una ambulancia al lugar.


#### ¿Qué hay que hacer?

1. *Project scoping*
2. EDA/GEDA
+ *Data profiling*:
  + Tabla de *data profiling* para las columnas numéricas. Explica las 3 cosas más relevantes.
  + Tabla de *data profiling* para las columnas categóricas. Explica las 3 cosas más relevantes.
  + Tabla de *data profiling* para las columnas de fecha. Explica las 3 cosas más relevantes.  
  + Tabla de *data profiling* para las columnas geoespaciales. Explica las 3 cosas más relevantes.
+ A lo más 10 gráficas que nos den *insights* importantes sobre la pregunta que queremos responder.
  + Al menos una de estas gráficas debe ser un mapa indicando los lugares en donde se han tenido reportes de incidentes viales -utiliza el parámetro `alpha`-, las llamadas que fueron incidentes viales reales serán de un color, e incidentes viales que no fueron reales -llamadas falsas- de otro.
  + Otra de estas gráficas debe mostrar la proporción de la etiqueta\* en el *data set*.
3. Transformación de variables: Fechas, numéricas, categóricas, etc.
4. Imputación de datos. En caso de que existan variables con datos faltantes, realiza una imputación "sencilla". Comenta cuáles son las implicaciones de tu decisión.
5. *Feature Engineering*
+ ¿Qué variables debemos eliminar porque no las podremos tener en el momento de la predicción?
+ Creación de variables
  + ¿Qué variables relevantes puedes crear?
+ Selección de variables
  + Generando un `RandomForest` utilizando `GridSearchCV` generando al menos 10 modelos diferentes -CV puede tener temporal CV- y utilizando eficiencia como la métrica de desempeño para seleccionar al "mejor modelo".
  + Elimina aquellas variables que aportan menos del 7% de información
  + ¿Cuáles variables son las que aportan más información para predecir? ¿De cuánto es la aportación de cada una?
  + ¿Cuáles son los hiperparámetros del mejor estimador?


\* Ver la sección `qué se entrega` -> `generate_label()`

#### ¿Qué se entrega?

A entregarse máximo el **24 de noviembre del 2020 23:59:59 CST**.

Liga al repositorio de github/gitlab/bit bucket que contiene el código de tu proyecto con:

+ En una carpeta `docs`:
  + Tu pdf de *scoping*.
  + Tu *slide deck* con las cosas más importantes de tu EDA/GEDA.
  + Un `html` con tus tablas de *profiling* y las explicaciones.
+ En una carpeta `notebooks`:
  + Tu `ipynb` con el código para generar tus *data profilings* y tu EDA/GEDA
+ Tu archivo `requirements.txt` -> lo ocuparé para poder correr sus procesos.
+ Un *script* `utils.py` que tenga las siguientes funciones:
    + `load_df(path)`: Recibe el *path* en donde se encuentra el `pickle` que se quiere volver a cargar.
    + `save_df(df, path)`: Recibe el `pickle` y el *path* en donde se debe guardar.
+ Un *script* `ingestion.py` que tenga las siguientes funciones:
   + `ingest_file(file_name)`: Recibe un *path* en donde se encuentra el CSV con los datos de reportes de incidentes viales. Regresa un *data frame* con los datos de los incidentes viales.
   + `drop_cols(df)`: Elimina las variables que no ocuparemos.
   + `generate_label()`: Crea en el *data frame* de los datos la variable `label` que es `1` cuando el código de cierre es `(F)` o `(N)`, `0` en otro caso.
   + `save_ingestion(df, path)`: Guarda en formato `pickle` (ver notebook `feature_engineering.ipynb`) el *data frame* que ya no tiene las columnas que ocuparemos y que incluye el `label` generado. El `pickle` se debe llamar `ingest_df.pkl` y se debe guardar en la carpeta `output`.
+ Un *script* `transformation.py` que tenga las siguientes funciones:
  + `load_ingestion(path)`: Recibe el *path* en donde se encuentra el pickle que generamos durante la ingestión.
  + `date_transformation(col, df)`: Recibe la columna que hay que transformar y el *data frame* al que pertenece.
  + `numeric_tranformation(col, df)`: Recibe la columna que hay que transformar y el *data frame* al que pertenece.
  + `categoric_trasformation(col, df)`: Recibe la columna que hay que transformar y el *data frame* al que pertenece.
  + `save_transformation(df, path)`: Guarda en formato `pickle` (ver notebook `feature_engineering.ipynb`) el *data frame* que ya tiene los datos transformados. El `pickle` se debe llamar `transformation_df.pkl` y se debe guardar en la carpeta `output`.
+ Un *script* `feature_engineering.py` que tenga las siguientes funciones como principales:
  + `load_transformation(path)`: Recibe el *path* en donde se encuentra el pickle que generamos durante la transformación.
  + `feature_generation(df)`: Recibe el *data frame* que contiene las variables a partir de las cuales crearemos nuevas variables. Estas nuevas variables se guardarán en este mismo *data frame*.
  + `feature_selection(df)`: Recibe el *data frame*  que contiene las variables de las cuales haremos una selección.
  + `save_fe(df, path)`: Guarda en formato `pickle` (ver notebook `feature_engineering.ipynb`) el *data frame* que ya tiene los *features* que ocuparemos. El `pickle` se debe llamar `fe_df.pkl` y se debe guardar en la carpeta `output`.
+ Un *script* `proyecto_1.py` que manda ejecutar las siguientes funciones:
  + `ingest(path)`: Función en el script `ingestion.py`
  + `transform()`: Función en el script `transformation.py`
  + `feature_engineering()`: Función en el script `feature_engineering.py`


**Estructura del proyecto**


+ data
    + **incidentes-viales-c5.csv**
+ docs
    + Data_Profiling.html
+ images
+ notebooks
    + EDA_GEDA_OFICIAL.ipynb
    + feature_engineering.ipynb
+ src
    + pipelines
        + feature_engineering.py
        + ingestion.py
        + transformation.py
    + utils
        + utils.py
    + proyect_1.py
+ requirements.txt

**Equipo**

|Nombre|Usuario Github| ID |
|------|--------------|----|
| Leonardo Ceja Pérez | lecepe00 | |
| Enrique Ortiz Casillas | EnriqueOrtiz27 | |
| Carolina Acosta Tovany | caroacostatovany | 000197627 |

**Para correr el proyecto**

+ Instalar los requirements.txt 
+ Poner el **archivo incidentes-viales-c5.csv** dentro del folder **data**
+ Ejecutar `python src/proyecto_1.py`
+ Si se corren los jupyter notebooks, previamente ejecutar en línea de comandos `export PYTHONPATH=$PWD`
