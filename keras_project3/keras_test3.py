import cv2 as cv
import keras
from keras import layers

# 1. Prepare the image for the Conv2D Layer
# Se carga la imagen con la librería de opencv y se lee la imagen en escala de grises
# Se carga la imagen con la librería de opencv y se lee la imagen en escala de grises
img = cv.imread("wallpaper-angemon.jpg")
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("imagen",img)
cv.imshow("imagen gris",img_gray)
cv.waitKey()
cv.destroyAllWindows()
# Se redimensiona la imagen, la redimensión se hace de 244x244 ya que es muy común
img = cv.resize(img_gray,(244,244))
height, width = img.shape
cv.imshow("imagen",img)
cv.waitKey()
cv.destroyAllWindows()
# Se crea el modleo secuencial y la capa de convolución
# Create a Secuential Model
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

model.add(layers.Conv2D(input_shape=(height,width,1),filters=64,kernel_size=(3,3)))
model.summary()

# Se entra más en detalle sobre los parámetros, en el caso que nos ocupa se muestra los datos de
# una sola capa

#Access layers parameters
# Method :: get_weights() devuelve una lista que consta de matrices NumPy.
# La primera matriz da los pesos de la capa y la segunda matriz da los sesgos.
filters, _ = model.layers[0].get_weights()
# Se tienen 64 imágenes de 3x3, que es el equivalente al kernel_size = (3,3) que es el tamaño
# de las imágenes donde se obtiene ñas caracteróstocas y se generan 64 filtros que es el
# equivalente a filters = 64
print(filters.shape)

# 2. We show the filters
# Se muestran los filtros que se han obtenidos anteriormente, en este caso concreto se hace
# redimensionando las imágenes
# Se normaliza los datos de la varaible filter
f_min, f_max = filters.min(), filters.max()
filters = (filters-f_min)/(f_max-f_min)
# Se muestran 10 filtros
for i in range(10):
    f = filters[:,:,:,i]
    f = cv.resize(f,(250,250),interpolation=cv.INTER_NEAREST)
    cv.imshow(str(i),f)
cv.waitKey()
cv.destroyAllWindows()