import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 150, 150
batch_size = 32

train_data = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = train_data.flow_from_directory('/kaggle/input/animals-detection-images-dataset/train', 
target_size = (img_height, img_width),batch_size=batch_size,class_mode='categorical',subset='training')

validation_generator = train_data.flow_from_directory('/kaggle/input/animals-detection-images-dataset/test',
target_size = (img_height, img_width),batch_size=batch_size,class_mode='categorical',subset='validation')

from tensorflow.keras import layers, models

model =models.Sequential([layers.Conv2D(32,(3,3), activation='relu',
input_shape=(img_height, img_width,3)),layers.MaxPooling2D(pool_size=(2,2)),
layers.Conv2D(64,(3,3), activation='relu'),layers.MaxPooling2D(pool_size=(2,2)) ,
layers.Conv2D(128,(3,3), activation='relu'),layers.MaxPooling2D(pool_size=(2,2)),
layers.Flatten(),layers.Dense(128, activation='relu'),
layers.Dense(len(train_generator.class_indices),activation='softmax')])
              
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])      

history = model.fit(train_generator, steps_per_epoch=train_generator.samples//batch_size,
validation_data=validation_generator, validation_steps=validation_generator.samples//batch_size,
epochs=10)

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy : {accuracy:.2f}')

import numpy as np
from tensorflow.keras.preprocessing import image
def predict_species(img_path):
    img=image.load_img(img_path,target_size=(img_height,img_width))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)/255.0
    predictions=model.predict(img_array)
    class_index=np.argmax(predictions)
    return list(train_generator.class_indices.keys())[class_index]

species=predict_species('/kaggle/input/animals-detection-images-dataset/test/Butterfly/00b34a3601c1398a.jpg')
print(f'This image is predicted to be: {species}')