import keras
from keras import layers
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. Load the image with OpenCV
# Se lee la imagen y se redimensiona para 244x244
img = cv.imread("wallpaper-angemon.jpg")
img = cv.resize(img,(224,224))
# Se separan los tres canales y se muestra por pantalla el resultado
b,g,r = cv.split(img)
cv.imshow("imagen",img)
cv.imshow("b",b)
cv.imshow("g",g)
cv.imshow("r",r)
cv.waitKey()
cv.destroyAllWindows()
 
# 2. Display feature map
# Se crea un modelo secuencial como se hace en las versiones anteriores
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
model.summary()

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
feature_map = model.predict(np.array([img]))

# Se muestran las 64 imágenes de los filtros que se aplican
for i in range(64):
	feature_img = feature_map[0,:,:,i]
	ax = plt.subplot(8,8,i+1)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.imshow(feature_img,cmap="gray")
plt.show()