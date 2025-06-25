import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Set paths
data_path = "C:/Users/david/OneDrive/Desktop/Mobile-price/Fake News Detection/"
save_path = data_path  # Save model to the same folder

# Load datasets
df_fake = pd.read_csv(os.path.join(data_path, "Fake.csv"), on_bad_lines='skip', engine='python')
df_real = pd.read_csv(os.path.join(data_path, "True.csv"), on_bad_lines='skip', engine='python')

# Label them
df_fake["label"] = "fake"
df_real["label"] = "real"

# Combine and clean
df = pd.concat([df_fake, df_real], ignore_index=True)[['text', 'label']].dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
with open(os.path.join(save_path, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(save_path, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"✅ Model and vectorizer saved to: {save_path}")
