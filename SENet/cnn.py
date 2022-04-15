import tensorflow
from keras.layers import Conv2D, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Activation, Reshape, MaxPooling2D, Input
from se import SEBlock


def _conv(filters = 32, kernel_size = (3,3), strides = 1, activation = "relu"):

  def f(input_x):

    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = (strides, strides), kernel_initializer="he_normal")(input_x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x

  return f

def _fc(units, activation = "relu"):

  def f(input_x):

    x = Dense(units = units, kernel_initializer="he_normal")(input_x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x
  return f

def CNN(input_shape = (32,32,3), output_units = 10, activation = 'sigmoid', SE = False, data_format = 'channels_last'):

  input_layer = Input(shape = input_shape)

  x = _conv(filters=32, activation = activation)(input_layer)
  x = SEBlock(se_ratio=1, activation = activation, data_format=data_format)(x) if SE == True else x
  x = _conv(filters=32, activation = activation)(x)
  x = MaxPooling2D(pool_size=(2,2))(x)

  x = SEBlock(se_ratio=1, activation = activation, data_format=data_format)(x) if SE == True else x
  x = _conv(filters=64, activation = activation)(x)
  x = MaxPooling2D(pool_size=(2,2))(x)

  x = SEBlock(se_ratio=1, activation = activation, data_format=data_format)(x) if SE == True else x
  x = _conv(filters=128, activation = activation)(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  x = Dropout(0.25)(x)

  #Use GlobalAveragePooling2D to replace flatten
  x = GlobalAveragePooling2D()(x)
  x = Reshape(1,1,x.shape[1])(x) if data_format == 'channels_first' else x
    
  x = _fc(units=256, activation = activation)(x)
  x = _fc(units=128, activation = activation)(x)
  x = Dropout(0.25)(x)
  output_layer = Dense(units = output_units, activation="softmax", kernel_initializer="he_normal")(x)

  model = tensorflow.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
  return model