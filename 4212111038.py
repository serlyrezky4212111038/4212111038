#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from skimage.feature import hog
from mlxtend.data import loadlocal_mnist


# In[4]:


images, labels = loadlocal_mnist(
    images_path='C:/Users/hp/Pictures/mnist-dataset/images/mnist-dataset/train-images-idx3-ubyte',
    labels_path='C:/Users/hp/Pictures/mnist-dataset/images/mnist-dataset/train-labels-idx1-ubyte'
)


# In[7]:


hog_features = [
    hog(image.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)[0]
    for image in images[:100]
]


# In[8]:


hog_features = np.array(hog_features)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(hog_features, labels[:100], test_size=0.2, random_state=42)


# In[21]:


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


# In[12]:


y_pred = svm_classifier.predict(X_test)


# In[22]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)

#generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# In[23]:


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:\n", conf_matrix)


# In[ ]:




