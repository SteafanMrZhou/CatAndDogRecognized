# @Author   : Steafan
# @Desc     : 
# @File     : CatAndDogRecognizeView.py
# @Create   : 2024-02-21 9:44

import os
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from django.http import HttpResponse
from django.shortcuts import render
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from PIL import Image
from keras.models import load_model

IMAGE_BASE_DIR = 'D://IT//MachineLearningPro//CatAndDogWebPro//CatAndDogWebPro//image//'
MODEL_TRAIN_IMAGE_BASE_DIR = 'D://IT//MachineLearningPro//CatAndDogWebPro//CatAndDogWebPro//web//data//train//'
MODEL_TEST_IMAGE_BASE_DIR = 'D://IT//MachineLearningPro//CatAndDogWebPro//CatAndDogWebPro//web//data//test//'

def index_render(request):
    html_val = {'title': 'testContent'}
    return render(request, 'index.html', context=html_val)


# 文件上传
def image_upload(request):
    if request.method == 'POST':
        image_file = request.FILES.get('img')
        image_path = os.path.join(IMAGE_BASE_DIR, image_file.name)
        f = open(image_path, 'wb')
        for chunk in image_file.chunks():
            f.write(chunk)
            f.close()

        return predict_model(request, image_path=image_path)
    else:
        return HttpResponse('error')


def generate_model(request):
    train_batch_size = 16
    image_width = 224
    image_height = 224
    image_channel_nums = 3
    train_image_data_generator = ImageDataGenerator(
        validation_split=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=15,
        preprocessing_function=preprocess_input,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_image_data_generator = ImageDataGenerator(
        validation_split=0.15,
        preprocessing_function=preprocess_input
    )
    test_image_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    train_image_generator = train_image_data_generator.flow_from_directory(
        MODEL_TRAIN_IMAGE_BASE_DIR,
        target_size=(image_width, image_height),
        seed=123,
        subset='training',
        shuffle=True,
        class_mode='categorical',
        batch_size=train_batch_size
    )
    validation_image_generator = validation_image_data_generator.flow_from_directory(
        MODEL_TRAIN_IMAGE_BASE_DIR,
        target_size=(image_width, image_height),
        seed=123,
        subset='validation',
        shuffle=False,
        class_mode='categorical',
        batch_size=train_batch_size
    )
    test_image_generator = test_image_data_generator.flow_from_directory(
        MODEL_TEST_IMAGE_BASE_DIR,
        target_size=(image_width, image_height),
        seed=123,
        shuffle=False,
        class_mode='categorical',
        batch_size=train_batch_size
    )
    train_image_samples = train_image_generator.samples
    validation_image_samples = validation_image_generator.samples
    test_image_samples = test_image_generator.samples
    train_classes = list(train_image_generator.class_indices.keys())
    num_classes = len(train_classes)
    physical_gpu_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_gpu_devices[0], True)
    with tf.device("/GPU:0"):
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(image_width, image_height, image_channel_nums)
        )
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(
            100,
            activation='relu'
        )(x)
        predictions = keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='random_uniform'
        )(x)
        model = keras.models.Model(
            inputs=base_model.input,
            outputs=predictions
        )
        for layer in base_model.layers:
            layer.trainable = False
        optimizer = keras.optimizers.Adam()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # 保存模型
        save_checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True,
                                                      verbose=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=True)
        history_model = model.fit(
            train_image_generator,
            steps_per_epoch=train_image_samples // 16,
            epochs=5,
            callbacks=[save_checkpoint, early_stopping],
            validation_data=validation_image_generator,
            verbose=True,
            validation_steps=validation_image_samples // 16
        )



def predict_model(request, image_path):
    # 加载模型
    loaded_model = load_model('model.h5')
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = loaded_model.predict(img_array)
    predicted_classes = np.argmax(predictions)
    predicted_probability = predictions[0][predicted_classes]
    # print(predicted_probability)
    class_labels = ['cat', 'dog']
    predicted_label = class_labels[predicted_classes]
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_label}\nProbability: {predicted_probability:.2%}')
    plt.show()
    predict_result = {
        "code": 0,
        "result_msg": str(predicted_label),
        "probability": str(predicted_probability)
    }
    return HttpResponse(json.dumps(predict_result))