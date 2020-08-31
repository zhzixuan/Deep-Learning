from keras import layers
from keras import models
from keras.applications import VGG16
import json
import os
import shutil
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


base_dir = '/userhome/34/zxzhao/assignment2/Datasets/cat_dog_car_bike/'
train_dir = os.path.join(base_dir, 'train') # /path/to/trainset
val_dir = os.path.join(base_dir, 'val')     # /path/to/validationset
test_dir = os.path.join(base_dir, 'test')   # /path/to/testset

mpl.use('Agg')

def predict(test_dir,
            output_dir="q5_result"):
    """
    Args:
        - test_dir: test set directory, note that there are four sub-directories under this directory, i.e., c0, c1, c2, c3.
        - output_dir: output directory
    Important: Your model shall be stored in ``q5_model`` directory. Hence, in this function, you have to implement:
        1. Restore your model from ``q5_model``; and
        2. Make predictions for test set using the restored model.
            Your results will be stored in file named ``prediction.json`` under ``q5_result``.
            Make sure your results are formatted as dictionary, e.g., {"image_file_name": "label"}. See prediction.json for reference.
            You can save the results with json.dump.
        3. You can define your model architecture in this scripts.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists("q5_model"):
        raise Exception("Model not found.")

    ### Restore your model
    model = models.load_model("./q5_model/q5_model.h5")
    
    ### Make predictions with restored model
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False)
    
    # print test accuracy
    test_loss, test_acc = model.evaluate_generator(test_generator)
    print('\nTest accuracy:', test_acc)
    
    # predict
    y_pre = model.predict_generator(test_generator) 
    
    ### Convert predicted results to dictionary
    predicted = np.argmax(y_pre, axis = 1)
    label = {0:'cat', 1:'dog', 2:'car', 3:'motorbike'}
    prediction = [label[i] for i in predicted]
    result = {}
    filenames = test_generator.filenames
    
    for i in range(len(filenames)):
        filename = filenames[i].split("/")[-1]
        result[filename] = prediction[i]
    print(result)
    
    ### Saved results to q5_result/prediction.json
    with open(os.path.join(output_dir,'prediction.json'), 'w') as f:
        json.dump(result, f, indent = 2)
    return True


def test_predict():
    shutil.rmtree("q5_result")
    predict(test_dir="/userhome/34/zxzhao/assignment2/Datasets/cat_dog_car_bike/test") # /path/to/testset
    result_path = "q5_result/prediction.json"
    if not os.path.isfile(result_path):
        raise FileNotFoundError()

    with open(result_path, encoding="utf-8") as fp:
        results = json.load(fp)

    assert isinstance(results, dict)

    print("\nPass.")
    

    
def train_val():
    # data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')    
    validation_generator = val_datagen.flow_from_directory(
        val_dir, 
        target_size=(224, 224), 
        batch_size=32, 
        class_mode='categorical')
    
    
    # load pre-trained model
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     conv_base.summary()
    
    # Freeze the layers
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
#         print(layer, layer.trainable)
    
    
    # Create the model    
    model = models.Sequential()
    model.add(conv_base)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    
   
    # Compile the model
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy', 
                  metrics=['acc'])
    
    # Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator, 
        validation_steps=50)
              
    # Save the model
    model.save('q5_model.h5')
    
    # Accuracy and loss plot
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
              
    epochs = range(1, len(acc) + 1)
              
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy') 
    plt.legend()
    plt.savefig('q5_acc.jpg')          
    plt.figure()             
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()  
    plt.savefig('q5_loss.jpg')
    


if __name__ == '__main__':
#     train_val()
    test_predict()

