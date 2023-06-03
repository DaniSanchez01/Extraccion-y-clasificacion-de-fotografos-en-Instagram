# Extracción y clasificación de fotógrafos en Instagram

Este repositorio se divide en dos partes bien diferenciadas. La primera (**extraccionInstagram**) aporta una infraestructura de recopilación de características para perfiles de Instagram y sus respectivas publicaciones, con ayuda de Instaloader. La segunda (**clasificacionFotografos**), a través de la información extraída ya estructurada como un conjunto de datos, creará un modelo de clasificación de perfiles, diferenciando los que pertenecen a fotógrafos profesionales y los que no. Para ejecutar cada parte es importante establecer como **directorio de trabajo** su carpeta correspondiente.

### Requisitos

Las siguientes librerías son necesarias para el funcionamiento de ambas partes.
  - Instaloader
  - TensorFlow
  - keras
  - caffee
  - Seaborn
  - Pandas
  - Sklearn

## 1. Infraestructura de descarga de perfiles y publicaciones

La carpeta **extraccionInstagram** aporta una infraestructura para la recopilación de información sobre perfiles y sus publicaciones, que estará dividida en dos partes. Al principio de cada código encontraremos una serie de **parámetros que deben ajustarse** para que el programa se adapte a nuestras necesidades.

### -Recopilación de perfiles 
El programa **recopiladorPerfiles** se encargará de, en base a un hashtag específico, recopilar los perfiles más recientes que lo hayan usado, guardándolos en un archivo JSON.
##### Parámetros a ajustar

- **accounts**: Nombres de usuario de las cuentas desde las que se harán las solicitudes al servidor.
- **passwords**: Claves de acceso de cada una de las cuentas especificadas.
- **userAgents**: Agentes de usuario que se establecerán para cada cuenta.
- **hashtag**: Nombre del hashtag a partir del cual se buscarán perfiles.
- **usernameFile**: Nombre del archivo donde se guardarán todos los perfiles recopilados.
- **profilesToDownload**: Máximo número de perfiles que se deben recopilar.
    


### -Extracción de características
Una vez se tenga la lista con todos los perfiles recopilados, el programa **descargaPerfiles** se encargará de extraer toda la información posible de cada uno. Se creará un archivo JSON con los datos más importantes de cada perfil y publicaciones asociadas, y se guardará información adicional de cada perfil en la **carpeta descargas**.

##### Parámetros a ajustar

- **accounts**: Nombres de usuario de las cuentas desde las que se harán las solicitudes al servidor.
- **passwords**: Claves de acceso a cada una de las cuentas especificadas.
- **userAgents**: Agentes de usuario que se establecerán para cada cuenta.
- **infoFile**: Nombre del archivo donde se guardará la información más importante de cada perfil y sus x primeras publicaciones.
- **photosToDownload**: Máximo número de publicaciones de las que debemos descargar datos en un perfil.
- **emailEmisor**: Dirección de correo electrónico con el que se mandarán alertas en caso de error.
- **emailPassword**: Contraseña de la cuenta de correo emisora
- **emailReceptor**: Dirección de correo electrónico que recibirá las alertas.

## 2. Clasificación de perfiles de fotógrafos profesionales

La carpeta **clasificacionFotografos** contiene código para crear un modelo de clasificación óptimo que permita distinguir los perfiles que pertenecen a fotógrafos profesionales y los que no. También se divide en dos partes.

### -Preprocesamiento

El código del archivo **preprocesamientoDatos** se encarga de hacer un preprocesamiento de un conjunto de datos. Este dividirá el código en un conjunto de entrenamiento y uno de validación, normalizará los datos, y aplicará una selección de características para descartar las que no sean útiles. Nos devolverá el conjunto de entrenamiento y el de test preprocesados y nos mostrará la diferencia de rendimiento del conjunto antiguo y los nuevos para la clasificación de Random Forest.

##### Parámetros a ajustar

- **datasetFile**: Nombre del archivo que contiene el conjunto de datos.

### -Clasificación

El programa **clasificacion** realizará un ajuste de hiperparámetros de modelos de clasificación con diferentes algoritmos de aprendizaje (LogisticRegression, Decision Trees, Random Forest y Gradient Boosting), eligiendo para cada uno de ellos los hiperparámetros que mejor rendimiento logren. Estos modelos estarán entrenados usando una validación cruzada de 5 pliegues y el conjunto de datos de entrenamiento. Tras esto, se hará una predicción usando el conjunto de validación y se mostrarán diferentes métricas y gráficas de los rendimientos de cada modelo. No hay parámetros a ajustar.
