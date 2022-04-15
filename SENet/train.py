import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import albumentations as albu
from albumentations import (HorizontalFlip, ShiftScaleRotate, GridDistortion)
from cnn import CNN
from setup import x_test,x_train,y_test,y_train

batch_size = 128
lr = 1e-3
epochs = 20
optimizer = tensorflow.keras.optimizers.Adam(lr=lr)
loss = "categorical_crossentropy"
metric = ['accuracy']

steps_per_epoch = len(x_train)//batch_size
validation_steps = len(x_test)//batch_size

def input_generator(x,y,aug,batch_size):

  x_len = len(x)
  batch_x, batch_y = [],[]
  while True:

    batch_indices = np.random.choice(x_len, size = batch_size)
    
    for idx in batch_indices:
      batch_y.append(y[idx])
      batch_x.append(aug(image = x[idx])['image']/255.0)

    batch_x, batch_y = np.stack(batch_x), np.stack(batch_y)
    yield batch_x, batch_y
    batch_x, batch_y = [],[]
  

aug_for_train = albu.Compose([HorizontalFlip(p=0.5),
                              ShiftScaleRotate(shift_limit=0.1,scale_limit=0.25,rotate_limit=20,p=0.5),
                              GridDistortion(p=0.5)])
aug_for_valid = albu.Compose([])

train_gen = input_generator(x_train, y_train, aug_for_train, batch_size)
valid_gen = input_generator(x_test, y_test, aug_for_valid, batch_size)

def display_training_result(history, SEFlag = False):

  plt.figure(figsize=(16,12))

  plt.subplot(2,1,1)
  plt.plot(history.history['loss'], label = 'train_loss', color = 'g')
  plt.plot(history.history['val_loss'], label = 'valid_loss', color = 'r')
  plt.title('Regular CNN training/validation loss' if not SEFlag else 'SENet CNN training/validation loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()

  plt.subplot(2,1,2)
  plt.plot(history.history['accuracy'], label = 'train_acc', color = 'g')
  plt.plot(history.history['val_accuracy'], label = 'valid_acc', color = 'r')
  plt.title('Regular CNN training/validation accuracy' if not SEFlag else 'SENet CNN training/validation accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend()

  plt.show()

# Build the regular Convolution neural network
regular_cnn = CNN()
regular_cnn.compile(loss = loss, metrics = metric, optimizer = optimizer)
regular_cnn.summary()

regular_cnn_history = regular_cnn.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                                                validation_data=valid_gen, validation_steps=validation_steps)

# Build the convolution neural network with SE block in it
SE_cnn = CNN(SE = True)
SE_cnn.compile(loss = loss, metrics = metric, optimizer = optimizer)
SE_cnn.summary()

SE_cnn_history = SE_cnn.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                                                validation_data=valid_gen, validation_steps=validation_steps)

display_training_result(regular_cnn_history)
display_training_result(SE_cnn_history, SEFlag = True)

regular_cnn_scores = regular_cnn.evaluate(x=x_test/255.0, y=y_test)
SE_cnn_scores = SE_cnn.evaluate(x=x_test/255.0, y=y_test)

print(f' Regular CNN :  loss {regular_cnn_scores[0]}, accuracy {regular_cnn_scores[1]}')
print(f' SE CNN : loss {SE_cnn_scores[0]}, accruacy {SE_cnn_scores[1]}')