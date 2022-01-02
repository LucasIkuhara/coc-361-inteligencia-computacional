from model import Classificador
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def main():
    # Baseado em exemplos de
    # https://www.tensorflow.org/tutorials/images/cnn?hl=pt-br


    classificador = Classificador()

    classificador.model.compile(optimizer='adam',
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                    from_logits=True
                                ),
                                metrics=['accuracy'])


    history = classificador.model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(test_images, test_labels)
                                      )


if __name__ == '__main__':
    main()
