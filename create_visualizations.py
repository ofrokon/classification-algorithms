import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def save_plot(fig, filename):
    plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Generate a random 2-class classification problem
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

def plot_classifier(classifier, title):
    # Train the model
    classifier.fit(X_train, y_train)
    
    # Make predictions on the mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    ax.set_title(f"{title} Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    save_plot(fig, f"{title.lower().replace(' ', '_')}.png")
    
    print(f"Accuracy on test set ({title}): {classifier.score(X_test, y_test):.2f}")

# Logistic Regression
plot_classifier(LogisticRegression(), "Logistic Regression")

# K-Nearest Neighbors
plot_classifier(KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors")

# Support Vector Machine
plot_classifier(SVC(kernel='rbf'), "Support Vector Machine")

# Decision Tree
plot_classifier(DecisionTreeClassifier(random_state=42), "Decision Tree")

# Random Forest
plot_classifier(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")

print("All visualizations have been saved as PNG files.")