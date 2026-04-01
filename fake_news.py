import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Display first 5 rows
print(data.head())

# Split data into input and output
X = data['text']
y = data['label']

# Convert text into numerical data
vectorizer = TfidfVectorizer(stop_words='english')
X_vector = vectorizer.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

# Create model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Take user input
news = input("Enter news text: ")

# Convert input text
news_vector = vectorizer.transform([news])

# Predict result
prediction = model.predict(news_vector)

# Output result
if prediction[0] == 'FAKE':
    print("This news is FAKE ❌")
else:
    print("This news is REAL ✅")
