import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = pd.read_csv("/content/Iris.csv")

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]   # It will contain all the features like depal length,width,petal length etc
y = data.iloc[:, -1]    # It will contain the target variable i.e whether the flower is anyone of the categories


# Split the data into training and test sets (70% training & 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Here we have used a KNN Classifier because these algorithms can be trained on 
#labeled data (features and corresponding target variable) of the Iris flower dataset 
# and then used to predict the variable(target) of a new, previously unseen Iris flower.

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5) #hyper parameter
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

