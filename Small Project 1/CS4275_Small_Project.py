import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Load the dataset
data = pd.read_csv('wine.csv')

# Assume 'Alcohol' and 'acid' are two columns in the dataset, and 'target' is the label column.
X = data[['Alcohol', 'Malic.acid']]
y = data['Wine']

# Split dataset into 70% training and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Helper function to plot decision regions
def plot_decision_regions(X, y, classifier, title, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Define the decision region
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predict
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot the decision surface
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

    plt.xlabel('Alcohol')
    plt.ylabel('Malic Acid')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


# Logistic Regression with different C values
C_values = [10, 100, 1000, 5000]

for C in C_values:
    model = LogisticRegression(C=C, random_state=42)
    model.fit(X_train, y_train)

    # Plot decision regions
    plot_decision_regions(X_train.values, y_train.values, model, title=f'Logistic Regression (C={C})')

# Plotting accuracy_score for different C values (Logistic Regression)
C_values = [10 ** x for x in range(-4, 5)]  # C=10^x where x in [-4, -3, ..., 4]
accuracy_scores = []

for C in C_values:
    model = LogisticRegression(C=C, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot accuracy vs C values for Logistic Regression
plt.plot(C_values, accuracy_scores, marker='o')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C for Logistic Regression')
plt.legend(loc='upper right')
plt.show()

# SVM with RBF Kernel for two gamma values
gammas = [0.1, 10]

for gamma in gammas:
    model = SVC(kernel='rbf', gamma=gamma, random_state=42)
    model.fit(X_train, y_train)

    # Plot decision regions for SVM with RBF kernel
    plot_decision_regions(X_train.values, y_train.values, model, title=f'SVM with RBF kernel (gamma={gamma})')

# Plotting accuracy for SVM with RBF Kernel over different C values
for gamma in gammas:
    accuracy_scores_svm = []
    for C in C_values:
        model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores_svm.append(accuracy)

    # Plot accuracy vs C values for SVM
    plt.plot(C_values, accuracy_scores_svm, marker='o', label=f'gamma={gamma}')

plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C for SVM with RBF Kernel')
plt.legend(loc='upper right')
plt.show()