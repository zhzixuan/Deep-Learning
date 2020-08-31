from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def learning_rate_range():
    """Give proper lower bound and upper bound for
    proper learning rate"""
    # Lower and upper bounds
    #######
    lower_bound = 0.1 
    upper_bound = 1e-6
    #######
    return lower_bound, upper_bound


def learnign_rate_examples():
    """Give three examples for a bad, not bad, and very good learning rate
    """
    #######
    bad_larning_rate = 0.1
    not_bad_learning_rate = 1e-4
    good_learning_rate = 1e-3
    #######
    return bad_larning_rate, not_bad_learning_rate, good_learning_rate


def test_learning_rate_range():
    lower, upper = learning_rate_range()
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    print("\nPass.")


def test_learnign_rate_examples():
    bad_larning_rate, not_bad_learning_rate, good_learning_rate = learnign_rate_examples()
    assert isinstance(bad_larning_rate, float)
    assert isinstance(not_bad_learning_rate, float)
    assert isinstance(good_learning_rate, float)
    print("\nPass.")


def train(learning_rate):   
    # data processing
    base_dir = '/userhome/34/zxzhao/assignment2/Datasets/cat_dog_car_bike/'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical') 
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir, 
        target_size=(150, 150), 
        batch_size=20, 
        class_mode='categorical')
    
    # create model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizers.RMSprop(lr=learning_rate), 
                  metrics=['acc'])
    
    # train 
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator, 
        validation_steps=50)


if __name__ == '__main__':
#     lr =0.1 # train_acc: 0.2877, val_acc: 0.2844
#     lr = 1e-2 # train_acc: 0.2867, val_acc: 0.2925
#     lr = 1e-3 # train_acc: 0.9764, val_acc: 0.8372
#     lr = 1e-4 # train_acc: 0.9302, val_acc: 0.8342
#     lr = 1e-5 # train_acc: 0.7815, val_acc: 0.7357
#     lr = 1e-6 # train_acc: 0.4867, 0.5015
#     train(lr)
    test_learning_rate_range()
    test_learnign_rate_examples()