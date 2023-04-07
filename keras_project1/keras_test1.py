import cv2 as cv
from tensorflow import keras
from keras.layers import Dense

# 1. Load the image with OpenCV
# Se carga la imagen con la librería de opencv y se lee la imagen en escala de grises
img = cv.imread("1366_2000.jpg")
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("imagen",img)
cv.imshow("imagen gris",img_gray)
cv.waitKey()
cv.destroyAllWindows()

# 2. How to pass the image to the neural network
# Para crear el modelo de la red neuronal con keras, primero hay que crear la capa de entrada, para ellos
# se explica como pasa la información y que es lo que hacen imprimiendo las imágenes.
# Las imagenes se componen de tres matrices superpuesta si estan a color o una si está en blanco y negro.
# Sus valores oscilan entre 0 y 255, y cuando se trata de color es la composición del azñul, verde y rojo.

print("La imagen en formato BGR es:", img)
print("La imagen en escala de grises es:", img_gray)

# Se crea la capa de entrada que tiene que tener las mismas dimensiones que las imágenes que se van a
# pasar por la red. En el caso que nos ocupa es (823, 1366, 3).
height, width = img_gray.shape
print(img.shape)

# Keras model structure
# tf.keras.Input(shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None,
#   ragged=None, type_spec=None, **kwargs)
##############################################################################################################
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
input_layer = keras.Input(shape=(height,width))
# Se muestra por pantalla la capa de entrada
print("Input layer shape:",input_layer)

# 3. Create layers with Keras
# Para crear capas ocultas se tiene que crear con las capas Densas
# tf.keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
#   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#   kernel_constraint=None, bias_constraint=None, **kwargs)
##############################################################################################################
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
Layer_1 = Dense(64)(input_layer)
Layer_2 = Dense(32)(Layer_1)
output = Dense(2)(Layer_2)

# Se muestra por pantalla el summary del modelo
# tf.keras.Model(*args, **kwargs)
model = keras.Model(inputs=input_layer,outputs=output)
model.summary()


