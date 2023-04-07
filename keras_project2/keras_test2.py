import cv2 as cv
import numpy as np
import keras
from keras import layers

# 1. Basics of image preprocessing
# Para pasar la imagen al modelo secuencial de Keras primero debemos obtener algunos parámetros:
# alto, ancho y el número de canales que en este caso corresponde a tres porque la imagen está en
# formato BGR si se usa la librería opencv.

# Load and preprocess image
img = cv.imread("1366_2000.jpg")
cv.imshow("imagen",img)
cv.waitKey()
cv.destroyAllWindows()
height, width, channels = img.shape
print(f"Height: {height}, Width: {width}, Channels: {channels}")

# 2. What is the sequential model
# El modelo secuencial es menos flexible que la API funcional porque las capas deben insertarse una
# debajo de la otra y siempre se usa solo con el mismo modelo. Pero tiene la ventaja de ser más
# fácil de programar.

# Create a Secuential Model
# tf.keras.Sequential(layers=None, name=None)
################################################################################################
#   layers  =  Lista opcional de capas para agregar al modelo.
#   name = Nombre opcional para el modelo.
model = keras.Sequential()

# Se añaden las capas densas del modelo con el sistema secuencial

# tf.keras.Input(shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None,
#   ragged=None, type_spec=None, **kwargs)
################################################################################################
#   shape = Una tupla de forma (enteros), sin incluir el tamaño del lote. Por ejemplo,
#       shape=(32,)indica que la entrada esperada serán lotes de vectores de 32 dimensiones.
#       Los elementos de esta tupla pueden ser Ninguno; Los elementos 'Ninguno' representan
#       dimensiones en las que se desconoce la forma.
#   batch_size = tamaño de lote estático opcional (entero).
#   name = Una cadena de nombre opcional para la capa. Debe ser único en un modelo
#       (no reutilice el mismo nombre dos veces). Se generará automáticamente si no se proporciona.
#   dtype = El tipo de datos esperado por la entrada, como una cadena ( float32, float64, int32...)
#   sparse = Un valor booleano que especifica si el marcador de posición que se va a crear es escaso.
#       Solo uno de 'irregular' y 'escaso' puede ser Verdadero. Tenga en cuenta que, si sparsees Falso,
#       los tensores dispersos aún se pueden pasar a la entrada; se densificarán con un valor
#       predeterminado de 0.
#   tensor = Tensor existente opcional para envolver en la Inputcapa. Si se establece, la capa usará el tf.
#       TypeSpecde este tensor en lugar de crear un nuevo tensor de marcador de posición.
#   ragged = 	Un valor booleano que especifica si el marcador de posición que se va a crear es irregular.
#       Solo uno de 'irregular' y 'escaso' puede ser Verdadero. En este caso, los valores de 'Ninguno'
#       en el argumento 'forma' representan dimensiones irregulares. Para obtener más información
#       sobre RaggedTensors, consulte esta guía .
#   type_spec = Un tf.TypeSpecobjeto desde el que crear el marcador de posición de entrada.
#       Cuando se proporciona, todos los demás argumentos, excepto el nombre, deben ser Ninguno.
#   **kwargs = Soporte de argumentos en desuso. Soportes y . batch_shapebatch_input_shape

# El método add se compone de la siguiente manera
# Method :: add(layer)
#   layer = instancia de capa
# Los errores (Raises) que se pueden dar son:
#   TypeError = Si layerno es una instancia de capa.
#   ValueError = En caso de que el layerargumento no conozca su forma de entrada.
#   ValueError = En caso de que el layerargumento tenga múltiples tensores de salida, o
#       ya esté conectado en otro lugar (prohibido en los Sequentialmodelos).

# Para crear capas ocultas se tiene que crear con las capas Densas
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
model.add(layers.Input(shape=(height,width,channels)))
model.add(layers.Dense(32))
model.add(layers.Dense(16))
model.add(layers.Dense(2))

# Se pasa la imagen al modelo secuencial, y hay que pasarla como vector porque los modelos están
# diseñados para procesar lotes de imagenes, por eso se usa la función de array de la librería
# numpy
preprocessed_img = np.array(([img]))
result = model(preprocessed_img)
print(result)