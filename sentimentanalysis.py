from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Example data (replace with your actual data)
X = [
    "This product is great!",
    "The delivery was fast and efficient.",
    "I'm satisfied with my purchase.",
    "The item arrived damaged.",
    "Very disappointed with the service.",
    "Product quality is not up to the mark."
]
y = ['positive', 'positive', 'positive', 'negative', 'negative', 'negative']

# Vectorize text data using Bag-of-Words
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
