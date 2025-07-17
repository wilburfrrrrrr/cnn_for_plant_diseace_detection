import keras
import keras.utils
import tensorflow as tf
import numpy as np

def load_and_preprocess(image_path):
	img = keras.utils.load_img(image_path, target_size=(800, 534))
	img_array = keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	# img_array /= 255.

	return img_array

model = keras.models.load_model('../model/leaves.h5')

img_path = './images/powdery.jpg'

img_preprocessed = load_and_preprocess(img_path)
prediction = model.predict(img_preprocessed)

scores = tf.nn.softmax(prediction[0])

class_names = ['Healthy', 'Powdery', 'Rust']

class_predicted = format(class_names[np.argmax(scores)])
confidence = 100 * np.max(scores)

if class_predicted == 'Healthy':
	print(f'La planta no tiene ninguna enfermedad, con una confianza de {confidence}%')
elif class_predicted == 'Powdery':
	print(f'La planta tiene Mildiu, con una confianza de {confidence}%')
elif class_predicted == 'Rust':
	print(f'La planta tiene Roya, con una confianza de {confidence}%')
