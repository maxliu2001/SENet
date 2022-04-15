from keras.layers import Dense, GlobalAveragePooling2D, Activation, Reshape, Permute, multiply

def SEBlock(se_ratio = 16, activation = "relu", data_format = 'channels_last', kernel_i = "he_normal"):

    def f(input):

        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input.shape[channel_axis]

        #Squeeze operation
        x = GlobalAveragePooling2D()(input)
        x = Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = Dense(input_channels // se_ratio, kernel_initializer= kernel_i)(x)
        x = Activation(activation)(x)
        
        #Excitation operation
        x = Dense(input_channels, kernel_initializer=kernel_i, activation='sigmoid')(x)
        x = Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = multiply([input, x])

        return x

    return f
