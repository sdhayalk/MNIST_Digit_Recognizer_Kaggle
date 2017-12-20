''' 
references from:
	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_rotation
	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shear
	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shift
	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_zoom
'''

import tensorflow as tf
import numpy as np

def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
	augmented_image = []
	augmented_image_labels = []
	counter = 0

	for num in range (0, dataset.shape[0]):
		# image_tensor = tf.convert_to_tensor(image, np.float32)
		
		# original image:
		augmented_image.append(dataset[num])
		augmented_image_labels.append(dataset_labels[num])

		for i in range(0, augementation_factor):
			if use_random_rotation:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shear:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				# augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.1, 0, row_axis=0, col_axis=1, channel_axis=2))
				# augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0, 0.1, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			# if use_random_zoom:
				# augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], 10, row_axis=0, col_axis=1, channel_axis=2))
				# augmented_image_labels.append(dataset_labels[num])

		counter += 1
		if counter%100 == 0:
			print("Augemented upto ", counter)

	return np.array(augmented_image), np.array(augmented_image_labels)
	

