import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

fake = pd.read_csv("dataset/fake.csv", encoding='latin1')
real = pd.read_csv("dataset/real.csv", encoding='latin1')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

df['content'] = df['title'] + " " + df['text']

X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Accuracy:", model.score(X_test_vec, y_test))

os.makedirs("backend/model", exist_ok=True)

pickle.dump(model, open("backend/model/model.pkl", "wb"))
pickle.dump(vectorizer, open("backend/model/vectorizer.pkl", "wb"))

print("Model & vectorizer saved!")