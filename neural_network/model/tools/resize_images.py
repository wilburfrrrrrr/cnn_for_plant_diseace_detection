from PIL import Image
import os

def resize_images(input_dir, output_dir, size=(800, 534)):
	os.makedirs(output_dir, exist_ok=True)
	for root, dirs, files in os.walk(input_dir):
		for file in files:
			if file.endswith('.jpg'):
				image = Image.open(os.path.join(root, file))
				image = image.resize(size)
				image.save(os.path.join(output_dir, file))



if __name__ == '__main__':
	resize_images('../../../plants_archive/Train/Train/Healthy', '../../../plants_archive_resize/Train/Train/Healthy')
	resize_images('../../../plants_archive/Train/Train/Powdery', '../../../plants_archive_resize/Train/Train/Powdery')
	resize_images('../../../plants_archive/Train/Train/Rust', '../../../plants_archive_resize/Train/Train/Rust')
	resize_images('../../../plants_archive/Validation/Validation/Healthy', '../../../plants_archive_resize/Validation/Validation/Healthy')
	resize_images('../../../plants_archive/Validation/Validation/Powdery', '../../../plants_archive_resize/Validation/Validation/Powdery')
	resize_images('../../../plants_archive/Validation/Validation/Rust', '../../../plants_archive_resize/Validation/Validation/Rust')
	resize_images('../../../plants_archive/Test/Test/Healthy', '../../../plants_archive_resize/Test/Test/Healthy')
	resize_images('../../../plants_archive/Test/Test/Powdery', '../../../plants_archive_resize/Test/Test/Powdery')
	resize_images('../../../plants_archive/Test/Test/Rust', '../../../plants_archive_resize/Test/Test/Rust')