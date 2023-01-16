from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model('C:\\Users\\SHERIF\\OneDrive\\Documents\\Deep_Learning\\mnist_fashion.h5')

img = ("C:\\Users\\SHERIF\\OneDrive\\Downloads\\six.jpg",1)

img = img/255.0

img = img.reshape(len(img),28,28,1)

pred = model.predict(img)

np.argmax(pred)
