import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Classificador:
    def __init__(self) -> None:
        # Trechos extraídos de
        # https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br
        # e https://cnvrg.io/cnn-tensorflow/

        self.model = keras.Sequential([
                                keras.layers.Conv2D(32, (3, 3), padding='n', activation="relu",
                                                    input_shape=(32, 32, 1)),

                                keras.layers.MaxPooling2D((2, 2), strides=2),

                                keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
                                keras.layers.MaxPooling2D((2, 2), strides=2),

                                keras.layers.Flatten(),
                                keras.layers.Dense(100, activation="relu"),
                                keras.layers.Dense(10, activation="softmax")
                                        ])

        self.model.summary()

    def inference(self, input):
        return self.model.call(input)


# Manter o escopo global limpo
def main() -> None:
    Classificador()


if __name__ == "__main__":
    main()
