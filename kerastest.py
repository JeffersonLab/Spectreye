import matplotlib.pyplot as plt

import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline(scale=1)

# Get a set of three example images
images = [
    keras_ocr.tools.read(url) for url in [
        "images/test2.jpg"
    ]
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=plt.subplots(figsize=(20, 20))[0])

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), ncols=1, figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
