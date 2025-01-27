import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from collections import Counter

# loading data
data_dict = pickle.load(open('/content/data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# data shape
print(f"date shape: {data.shape}")
print(f"label shape: {labels.shape}")

# data dist
label_counts = Counter(labels)
print("label dist:", label_counts)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% accuracy')

# confusion matrix
print("confusion matrix:")
print(confusion_matrix(y_test, y_predict))

# save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)