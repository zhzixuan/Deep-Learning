from keras import layers
from keras import models
from keras import optimizers

def determine_k_value():
    """Return k value."""
    #######
    k = 4
    #######
    return k


def build_cnn_architecture():
    """Return instance of keras.models.Sequential."""
    #######
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(150, 150, 3), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='sigmoid', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='sigmoid', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(4, activation='softmax'))
    #######
    return model


def test_determine_k_value():
    k = determine_k_value()
    assert isinstance(k, int)
    print("\nPass.")


def test_build_cnn_architecture():
    model = build_cnn_architecture()
    assert isinstance(model, models.Sequential)
    print("\nPass.")


if __name__ == '__main__':
    test_determine_k_value()
    test_build_cnn_architecture()