import os
import zipfile
import random
import tensorflow as tf
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99.0 accuracy so cancelling training!")
      self.model.stop_training = True

def extract_archive(zip_path):
    zip_ref  = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall('res/data/')
    zip_ref.close()
    print(len(os.listdir('res/data/PetImages/Cat/')))
    print(len(os.listdir('res/data/PetImages/Dog/')))

def create_dataset_directories():
    try:
        os.mkdir('res/cats-v-dogs')
        os.mkdir('res/cats-v-dogs/training')
        os.mkdir('res/cats-v-dogs/testing')
        os.mkdir('res/cats-v-dogs/training/cats')
        os.mkdir('res/cats-v-dogs/training/dogs')
        os.mkdir('res/cats-v-dogs/testing/cats')
        os.mkdir('res/cats-v-dogs/testing/dogs')
    except OSError:
        pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE, BATCH_COEFF):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int((len(files) * SPLIT_SIZE)*BATCH_COEFF)
    testing_length = int((len(files) - training_length)*BATCH_COEFF)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)



def divide_data_between_directories(SPLIT_SIZE, BATCH_COEFF):

    CAT_SOURCE_DIR = "res/data/PetImages/Cat/"
    TRAINING_CATS_DIR = "res/cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "res/cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = "res/data/PetImages/Dog/"
    TRAINING_DOGS_DIR = "res/cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "res/cats-v-dogs/testing/dogs/"

    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, SPLIT_SIZE, BATCH_COEFF)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, SPLIT_SIZE, BATCH_COEFF)

    print(len(os.listdir('res/cats-v-dogs/training/cats/')))
    print(len(os.listdir('res/cats-v-dogs/training/dogs/')))
    print(len(os.listdir('res/cats-v-dogs/testing/cats/')))
    print(len(os.listdir('res/cats-v-dogs/testing/dogs/')))

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])


def draw_charts(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs=range(len(acc)) # Get number of epochs

    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()

    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.show()

def save_model(model, filename):
    model.save(filename)

#----------------------------------------------------------

def main():

    extract_archive('res/cats-and-dogs.zip')
    create_dataset_directories()
    divide_data_between_directories(SPLIT_SIZE = 0.8, BATCH_COEFF = 0.01)
    
    model = create_model()
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])


    TRAINING_DIR = "res/cats-v-dogs/training/"
    train_datagen = ImageDataGenerator(rescale=1./255,
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

    VALIDATION_DIR = "res/cats-v-dogs/testing/"
    validation_datagen = ImageDataGenerator(rescale=1./255,
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=10,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

    history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)

    draw_charts(history)

    save_model(model, 'my_model')


#------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------