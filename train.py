import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load data from the DVC-tracked file
print("Loading data...")
df = pd.read_csv('data/iris.csv')

# 2. Prepare data for training
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
print("Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Save the model locally
joblib.dump(model, 'models/model.joblib')
print("Model saved to models/model.joblib")
