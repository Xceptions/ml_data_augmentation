"""
    Write a function that can shift an MNIST image in any direction (left, right, up, or down)
    by one pixel.
    Then, for each image in the training set, create four shifted copies (one per direction) and
    add them to the training set. Finally, train your best model on this expanded training set and
    measure its accuracy on the test set. You should observe that your model performs even better
    now! This technique of artificially growing the training set is called data augmentation or
    training set expansion.

    answer gotten = 0.9988571428571429
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift as img_shift


mnist = fetch_mldata('MNIST original')

# reduce the mnist data to (1000, 100) samples
"""
    Create four copies to hold the new shifted forms
"""

X, x_discard, y, y_discard = train_test_split(
    mnist.data, mnist.target, random_state=0, test_size=0.95
)



X1 = X
X2 = X
X3 = X
X4 = X
y1 = y
y2 = y
y3 = y
y4 = y



class ImgManipulation():
    """
        this class shifts an image 1px to a specified direction
        using scipy
    """

    def __init__(self, main_data):
        self.main_data = main_data
        self.image = []
        self.shiftedImage = []

    def reshapeData(self):
        self.image = [x.reshape(28, 28) for x in self.main_data]
        return self

    def shiftImage(self, direction):
        if direction == 'right':
            self.shiftedImage = [img_shift(x, [0, 1], output=x, cval=0) for x in self.image]
        elif direction == 'left':
            self.shiftedImage = [img_shift(x, [0, -1], output=x, cval=0) for x in self.image]
        elif direction == 'up':
            self.shiftedImage = [img_shift(x, [-1, 0], output=x, cval=0) for x in self.image]
        elif direction == 'down':
            self.shiftedImage = [img_shift(x, [1, 0], output=x, cval=0) for x in self.image]
        else:
            return
        return self


    def returnImage(self):
        self.shiftedImage = [x.reshape(1, 784) for x in self.shiftedImage]
        return self.shiftedImage
        

"""
    Iterate the training data over ImgManipulation class and retrieve the shifted forms
"""

ImgManipulation(X1).reshapeData().shiftImage("left").returnImage()
ImgManipulation(X2).reshapeData().shiftImage("right").returnImage()
ImgManipulation(X3).reshapeData().shiftImage("up").returnImage()
ImgManipulation(X4).reshapeData().shiftImage("down").returnImage()


shiftedCopiesX = np.append(X1, X2, axis=0)
shiftedCopiesX = np.append(shiftedCopiesX, X3, axis=0)
shiftedCopiesX = np.append(shiftedCopiesX, X4, axis=0)

shiftedCopiesy = np.append(y1, y2, axis=0)
shiftedCopiesy = np.append(shiftedCopiesy, y3, axis=0)
shiftedCopiesy = np.append(shiftedCopiesy, y4, axis=0)

new_data_x = np.append(X1, shiftedCopiesX, axis=0)
new_data_y = np.append(y1, shiftedCopiesy, axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    new_data_x, new_data_y, random_state=0
)

"""
 variable contains the new set to be trained,
 now to apply KNeighbors on the training set
"""

clf_knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
clf_knn.fit(X_train, y_train)
clf_knn.predict(X_test)
print(clf_knn.score(X_test, y_test))
