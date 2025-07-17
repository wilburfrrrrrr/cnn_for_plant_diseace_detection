import tensorflow as tf
from keras import layers, models
# from keras.optimizers import adam
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

class DiceaseDetector:

	def __init__(self, batch_size = 16, epochs = 20, img_height = 800, img_width = 534, num_classes = 3):
		self.batch_size = batch_size
		self.epochs = epochs
		self.img_height = img_height
		self.img_width = img_width
		self.num_classes = num_classes
		self.model = self.build_model()

	def build_model(self):
		return models.Sequential([
				layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
				layers.BatchNormalization(),
				layers.MaxPooling2D((2, 2)),
				layers.Conv2D(32, (3, 3), activation='relu'),
				layers.BatchNormalization(),
				layers.MaxPooling2D((2, 2)),
				layers.Conv2D(64, (3, 3), activation='relu'),
				layers.BatchNormalization(),
				layers.MaxPooling2D((2, 2)),		
				layers.Flatten(),
				layers.Dense(64, activation='relu'),
				layers.Dropout(0.25),
				layers.Dense(self.num_classes, activation='softmax')
			])
	
	def train_generator(self):
		return image_dataset_from_directory(
			'../../plants_archive_resize/Train/Train',
			labels='inferred',
			label_mode='categorical',
			color_mode='rgb',
			batch_size=self.batch_size,
			image_size=(self.img_height, self.img_width),
			shuffle=True,
			seed=123,
			validation_split=0.2,
			subset='training'
		)
	
	def validation_generator(self):
		return image_dataset_from_directory(
			'../../plants_archive_resize/Validation/Validation',
			labels='inferred',
			label_mode='categorical',
			color_mode='rgb',
			batch_size=self.batch_size,
			image_size=(self.img_height, self.img_width),
			shuffle=True,
			seed=123,
			validation_split=0.2,
			subset='validation'
		)
	
	def test_generator(self):
		return image_dataset_from_directory(
			'../../plants_archive_resize/Test/Test',
			labels='inferred',
			label_mode='categorical',
			color_mode='rgb',
			batch_size=self.batch_size,
			image_size=(self.img_height, self.img_width),
			shuffle=False
		)
	
	def compile_model(self):
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	def get_history(self):
		return self.model.fit(
			self.train_generator(),
			validation_data=self.validation_generator(),
			epochs=self.epochs
		)
		
	def save_model(self):
		self.model.save('leaves.h5')
	
	def get_evaluation(self):
		return self.model.evaluate(self.test_generator())

	def train(self):
		history = self.get_history()
		test_loss, test_acc = self.get_evaluation()
		print(f'Test accuracy: {test_acc}')
		print(f'Test loss: {test_loss}')
		print(f'history: {history.history}')
		self.plot_history(history)
		self.save_model()

	def plot_history(self, history):
		plt.plot(history.history['accuracy'], label='accuracy')
		plt.plot(history.history['val_accuracy'], label='val_accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend(loc='lower right')
		plt.savefig('accuracy.png')
		plt.show()
		
if __name__ == '__main__':
	dd = DiceaseDetector()	
	dd.compile_model()
	dd.train()
	print(f"{dd.model.summary()}")
