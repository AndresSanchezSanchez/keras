import cv2 as cv
import keras
from keras import layers
import numpy as np
import tensorflow as tf

# 1. What is the Max pooling layer?
# Esta operación consiste en tomar un cuadrado de píxeles vecinos y tomar el valor máximo
# de todos esos píxeles que están en el cuadrado. También reduce la resolución de
# la imagen principal sacando una serie de características.
# Se lee la imagen que se va a tratar con la librería de opencv y redimensiona a 32x32
img = cv.imread("wallpaper-angemon.jpg")
img = cv.resize(img,(224,224))
cv.imshow("imagen",img)
cv.waitKey()
cv.destroyAllWindows()

# Se crea una semilla para generar aleatoriamente filtros en función de esa semilla
# Las operaciones que se basan en una semilla aleatoria en realidad la derivan de 
# dos semillas: las semillas globales y de nivel de operación.
# Esto establece la semilla global.
# Sus interacciones con semillas a nivel de operación son las siguientes:
# 1. Si no se establece la semilla global ni la semilla de la operación: se 
#	utiliza una semilla seleccionada al azar para esta operación.
# 2. Si se establece la inicialización global, pero no la inicialización de 
#	la operación: el sistema elige de manera determinista una inicialización de 
#	la operación junto con la inicialización global para que obtenga una secuencia 
#	aleatoria única. Dentro de la misma versión de tensorflow y código de usuario, 
#	esta secuencia es determinista. Sin embargo, en diferentes versiones, esta secuencia 
#	puede cambiar. Si el código depende de semillas particulares para funcionar, 
#	especifique semillas globales y de nivel de operación explícitamente.
# 3. Si se establece la inicialización de la operación, pero no la inicialización global: 
#	se utiliza una inicialización global predeterminada y la inicialización de la operación 
#	especificada para determinar la secuencia aleatoria.
# 4. Si se establecen tanto la semilla global como la operación: ambas semillas se utilizan 
#	conjuntamente para determinar la secuencia aleatoria.

# tf.random.set_seed(seed)
################################################################################################
#	seed = entero
tf.random.set_seed(0)

# 2. CNN Model
# Se crea el modelo de la red neuronal convolucional. Se crea un modelo secuencial, una capa
# convolucional y una capa de maxPooling

# tf.keras.Sequential(layers=None, name=None)
################################################################################################
#   layers  =  Lista opcional de capas para agregar al modelo.
#   name = Nombre opcional para el modelo.
model = keras.Sequential()
# El método add se compone de la siguiente manera
# Method :: add(layer)
#   layer = instancia de capa
# Los errores (Raises) que se pueden dar son:
#   TypeError = Si layerno es una instancia de capa.
#   ValueError = En caso de que el layerargumento no conozca su forma de entrada.
#   ValueError = En caso de que el layerargumento tenga múltiples tensores de salida, o
#       ya esté conectado en otro lugar (prohibido en los Sequentialmodelos).

# tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
#   dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
#   kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
#   bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
#   **kwargs)
################################################################################################
#   filters = Entero, la dimensionalidad del espacio de salida (es decir, el número de filtros de
#       salida en la convolución).
#   kernel_size = Un entero o tupla/lista de 2 enteros, especificando la altura y el ancho de la
#       ventana de convolución 2D. Puede ser un entero único para especificar el mismo valor
#       para todas las dimensiones espaciales.
#   strides = Un entero o tupla/lista de 2 enteros, especificando los pasos de la convolución a
#       lo largo de la altura y el ancho. Puede ser un entero único para especificar el mismo
#       valor para todas las dimensiones espaciales. Especificar cualquier valor de
#       zancada != 1 es incompatible con especificar cualquier valor != 1. dilation_rate
#   padding = uno de "valid"o "same"(no distingue entre mayúsculas y minúsculas). "valid"significa
#       sin relleno. "same"da como resultado un relleno con ceros uniformemente a la
#       izquierda/derecha o arriba/abajo de la entrada. Cuando padding="same"y strides=1,
#       la salida tiene el mismo tamaño que la entrada.
#   data_format =Una cadena, una de (predeterminado) o . El orden de las dimensiones en las
#       entradas. corresponde a entradas con forma mientras que corresponde a entradas con forma.
#       El valor predeterminado es el que se encuentra en el archivo de configuración de Keras en.
#       Si nunca lo establece, entonces será.
#       channels_lastchannels_firstchannels_last(batch_size, height, width, channels)
#       channels_first(batch_size, channels,height, width)
#       image_data_format~/.keras/keras.jsonchannels_last
#   dilation_rate = un número entero o tupla/lista de 2 números enteros, especificando la
#       tasa de dilatación a usar para la convolución dilatada. Puede ser un entero único
#       para especificar el mismo valor para todas las dimensiones espaciales. Actualmente,
#       especificar cualquier valor != 1 es incompatible con especificar cualquier valor de
#       zancada != 1. dilation_rate
#   groups =Un entero positivo que especifica el número de grupos en los que se divide la entrada
#       a lo largo del eje del canal. Cada grupo se convoluciona por separado con filtros.
#       La salida es la concatenación de todos los resultados a lo largo del eje del canal.
#       Los canales de entrada y ambos deben ser divisibles por. filters / groupsgroupsfiltersgroups
#   activation = Función de activación a utilizar. Si no especifica nada, no se aplica ninguna
#       activación (ver keras.activations).
#   use_bias = Booleano, si la capa usa un vector de sesgo.
#   kernel_initializer = Inicializador para la kernelmatriz de pesos (ver keras.initializers).
#       El valor predeterminado es 'glorot_uniform'.
#   bias_initializer = Inicializador para el vector de polarización (ver keras.initializers).
#       El valor predeterminado es 'ceros'.
#   kernel_regularizer = Función de regularización aplicada a la kernelmatriz de pesos
#       (ver keras.regularizers).
#   bias_regularizer = Función de regularizador aplicada al vector de polarización
#       (ver keras.regularizers).
#   activity_regularizer = Función de regularizador aplicada a la salida de la capa
#       (su "activación") (ver keras.regularizers).
#   kernel_constraint = Función de restricción aplicada a la matriz kernel (ver keras.constraints).
#   bias_constraint = Función de restricción aplicada al vector de polarización
#       (ver keras.constraints).
model.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3)))

# tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', 
#	data_format=None, **kwargs)
################################################################################################
#	pool_size = entero o tupla de 2 enteros, tamaño de la ventana sobre la que tomar el máximo. 
#		tomará el valor máximo en una ventana de agrupación de 2x2. Si solo se especifica 
#		un número entero, se utilizará la misma longitud de ventana para ambas dimensiones. (2, 2)
#	strides = Entero, tupla de 2 enteros o Ninguno. Valores de zancadas. Especifica cuánto 
#		se mueve la ventana de agrupación para cada paso de agrupación. Si es Ninguno, se 
#		establecerá de forma predeterminada en . pool_size
#	padding = Uno de "valid"o "same"(no distingue entre mayúsculas y minúsculas). 
#		"valid"significa sin relleno. "same"da como resultado un relleno uniforme hacia la 
#		izquierda/derecha o hacia arriba/abajo de la entrada, de modo que la salida tiene la 
#		misma dimensión de alto/ancho que la entrada.
#	data_format = Una cadena, una de (predeterminado) o. El orden de las dimensiones en 
#		las entradas. corresponde a entradas con forma mientras que corresponde a entradas 
#		con forma. El valor predeterminado es el que se encuentra en el archivo de configuración 
#		de Keras en. Si nunca lo configura, entonces será "channels_last". 
#		channels_lastchannels_firstchannels_last(batch, height, width, channels)
#		channels_first(batch, channels, height, width)image_data_format~/.keras/keras.json
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# tf.keras.layers.Flatten(data_format=None, **kwargs)
################################################################################################
#	data_format = Una cadena, una de (predeterminado) o . El orden de las dimensiones 
#	en las entradas. corresponde a entradas con forma mientras que corresponde a 
#	entradas con forma. El valor predeterminado es el que se encuentra en el archivo de 
#	configuración de Keras en . Si nunca lo configura, entonces será "channels_last". 
#	channels_lastchannels_firstchannels_last(batch, ..., channels)
#	channels_first(batch, channels, ...)image_data_format~/.keras/keras.json
model.add(layers.Flatten())

# tf.keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
#   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#   kernel_constraint=None, bias_constraint=None, **kwargs)
################################################################################################
#   units = Entero positivo, dimensionalidad del espacio de salida.
#   activation = Función de activación a utilizar. Si no especifica nada, no se aplica ninguna activación
#       (es decir, activación "lineal": a(x) = x).
#   use_bias = Booleano, si la capa usa un vector de sesgo.
#   kernel_initializer = Inicializador para la matriz de pesos del núcleo.
#   bias_initializer = Inicializador para el vector de polarización.
#   kernel_regularizer = Inicializador para el vector de polarización.
#   bias_regularizer = Función de regularizador aplicada al vector de polarización.
#   activity_regularizer = Función de regularizador aplicada a la salida de la capa (su "activación").
#   kernel_constraint = Función de restricción aplicada a la matriz de pesos del núcleo.
#   bias_constraint = Función de restricción aplicada al vector de polarización.
model.add(layers.Dense(units=10))

# Con la función predict se obtiene el resultado con la alicación de una capa Flatten
# Se crea el modelo predictivo con el método predict
# predict(x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, 
# 	workers=1, use_multiprocessing=False)
################################################################################################
#	x = Muestras de entrada. Podría ser:
#		Una matriz Numpy (o similar a una matriz) o una lista de matrices (en caso de que el modelo 
#			tenga múltiples entradas).
#		Un tensor de TensorFlow, o una lista de tensores (en caso de que el modelo tenga 
#			múltiples entradas).
#		Un conjunto de tf.datadatos.
#		Un generador o keras.utils.Sequenceinstancia. En la Unpacking behavior for iterator-like 
#			inputssección de Model.fit
# 	batch_size = entero o None. Número de muestras por lote. Si no se especifica, el valor 
#		predeterminado es 32. No especifique si sus datos están en forma de conjuntos de datos, 
#		generadores o instancias (ya que generan lotes). batch_sizebatch_sizekeras.utils.Sequence
# 	verbose = "auto", 0, 1 o 2. Modo de verbosidad. 0 = silencioso, 1 = barra de progreso, 2 = línea única. 
#		"auto"el valor predeterminado es 1 para la mayoría de los casos y 2 cuando se usa con 
#		ParameterServerStrategy. Tenga en cuenta que la barra de progreso no es particularmente útil cuando 
#		se registra en un archivo, por lo que verbose=2se recomienda cuando no se ejecuta de forma 
#		interactiva (por ejemplo, en un entorno de producción).
# 	steps = Número total de pasos (lotes de muestras) antes de declarar finalizada la ronda de predicción. 
# 		Ignorado con el valor predeterminado de None. Si x es un conjunto de tf.data datos y stepses 
#		Ninguno, predict()se ejecutará hasta que se agote el conjunto de datos de entrada.
# 	callbacks = Lista de keras.callbacks.Callbackinstancias. Lista de devoluciones de llamada para aplicar 
#		durante la predicción. Ver devoluciones de llamada .
# 	max_queue_size = Entero. Usado solo para generador o keras.utils.Sequence entrada. Tamaño máximo para 
#		la cola del generador. Si no se especifica, el valor predeterminado será 10. max_queue_size
# 	workers = Entero. Usado solo para generador o keras.utils.Sequenceentrada. Número máximo de procesos 
#		para activar cuando se utiliza subprocesos basados ​​en procesos. Si no se especifica, 
#		workersel valor predeterminado será 1.
# 	use_multiprocessing = booleano. Usado solo para generador o keras.utils.Sequenceentrada. Si True, 
#		utilice subprocesos basados ​​en procesos. Si no se especifica, el valor 
#		predeterminado será. Tenga en cuenta que debido a que esta implementación se basa en el 
#		multiprocesamiento, no debe pasar argumentos no seleccionables al generador, ya que no se pueden 
#		pasar fácilmente a los procesos secundarios. use_multiprocessingFalse
result = model.predict(np.array([img]))
print("################################################################################################")
print("################################################################################################")
print("################################################################################################")
print(result.shape)
print(result)

cv.imshow("img",result)
cv.waitKey()
cv.destroyAllWindows()