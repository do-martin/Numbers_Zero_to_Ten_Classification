import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    from keras.preprocessing import image

    # 28x28 images of hand-written digits 0-9
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    """28 x 28 dimensionales Array"""
    # print(x_train[0])
    # print(len(x_train[0]))

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=3)

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    """Visualize"""

    plt.imshow(x_train[0])
    # plt.show()
    # print(x_train[0])

    # model.save('epic_num_reader.model')

    new_model = tf.keras.models.load_model('epic_num_reader.model')

    predictions = new_model.predict([x_test])
    """print(predictions[0])

    print(np.argmax(predictions[0]))
    """
    # plt.imshow(x_test[0])
    # plt.show()

    """

    for i in predictions[0]:
        print('Prediction is : ' + str(i))"""

    """img = image.load_img('number_1.png', target_size=(28,28))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

    img = np.asarray(img)
    # img = tf.keras.utils.normalize(img, axis=1)
    tf.reshape(img, (1, 28, 28, 1))
    predicted = new_model.predict(img)
    # print(np.argmax(predicted[0]))
    """

    # Load the image
    img = Image.open('number_1.png').convert('L')

    # Resize to 28x28
    img = img.resize((28, 28))
    plt.imshow(img)
    plt.show()

    # Convert image to numpy array
    img = np.array(img)

    # Invert the image colors (optional, depending on your image)
    img = 255 - img

    # Normalize the image to 0-1 values
    img = img / 255.0

    # Reshape the image to match the model's input shape
    img = img.reshape(1, 28, 28)

    # Predict the digit
    predicted = new_model.predict(img)

    # Print the predicted digit
    print(np.argmax(predicted[0]))
    print(predicted)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
