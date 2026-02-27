import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("C:\\Users\\DELL\\OneDrive\\Desktop\\MUJ\\Deep Learning\\lab\\lab 4\\lenet.keras")

FOLDER = "C:\\Users\\DELL\\OneDrive\\Desktop\\MUJ\\Deep Learning\\lab\\lab 4\\images"

correct = 0
total = 0

for file in os.listdir(FOLDER):
    if not file.endswith(".png"):
        continue

    true_label = int(file.split(".")[0])  
    path = os.path.join(FOLDER, file)

    img = Image.open(path).convert("L")
    img = img.resize((28, 28))

    img = np.array(img)
    img = 255 - img
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img, verbose=0)
    pred_label = pred.argmax()
    confidence = pred.max()
    
    correct += (pred_label == true_label)
    total += 1

    plt.imshow(img[0, :, :, 0], cmap="gray")
    plt.title(
        f"Pred: {pred_label} ({confidence:.2f}) | True: {true_label}"
    )
    plt.axis("off")
    plt.show()

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")