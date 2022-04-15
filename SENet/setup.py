from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2, random_state = 2019)