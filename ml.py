import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import random

digits= load_digits()

x=digits.data
y=digits.target

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

model = MLPClassifier(hidden_layer_sizes=90,max_iter=20000)
model.fit(X_train, y_train)

index = random.randint(0, len(X_test)-1 )  
image = X_test[index]
true_label = y_test[index]

prediction = model.predict([image])[0]

plt.imshow(image.reshape(8, 8), cmap='gray')
plt.title(f"True: {true_label} | Predicted: {prediction}")
plt.show()
