from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

base_dir = '/userhome/34/zxzhao/assignment2/Datasets/cat_dog_car_bike/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

mpl.use('Agg')
    
def transfer_learning_with_vggnet():
    """Return model built on pre-trained VGG16."""
    #######
    # load pre-trained model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#     vgg_conv.summary()
    
    # Freeze the layers
    for layer in vgg_conv.layers:
        layer.trainable = False
#         print(layer, layer.trainable)

    # Create the model    
    model = models.Sequential()
    
    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
#     model.summary()
    #######
    return model


def test_transfer_learning_with_vggnet():
    model = transfer_learning_with_vggnet()
    assert isinstance(model, models.Sequential)
    print("\nPass.")
    return model


if __name__ == '__main__':
    
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
    
    # build model
    model = test_transfer_learning_with_vggnet()
              
    # Compile the model
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='categorical_crossentropy', 
                  metrics=['acc'])
    
    # Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator, 
        validation_steps=50)
              
#     # Save the model
#     model.save('vgg16.h5')
    
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
    
    plt.savefig('q4_acc.jpg')
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
              
#     plt.show()
    plt.savefig('q4_loss.jpg')
    
