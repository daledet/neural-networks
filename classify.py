import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape)
print(test_images.shape)

# Build model with 3 layers 2 layers with 64 neurons and the relu function - 1 layer 10 nuerons and softmax function
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=784))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32
)

# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

# Predict on first 5 test images
predictions = model.predict(test_images[:5])
print(np.argmax(predictions, axis=1))
print(test_labels[:5])

for i in range(0, 5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()
