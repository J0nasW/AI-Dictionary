import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import joblib, time
import numpy as np
from ast import literal_eval

print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print("")

# Load your data
df = pd.read_csv('data/dictionaries/core_keywords.csv')

# Get the minimum number of rows per source and sample the data
min_rows = df['source'].value_counts().min()
df = df.groupby('source').apply(lambda x: x.sample(min_rows, random_state=42)).reset_index(drop=True)
print(f"Sampled {min_rows} rows per source.")

X = df["embedding"].apply(literal_eval).tolist()
y = df['source'].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights to handle imbalances
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define the SVM Classifier
svm = SVC()

# Grid Search Parameters
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto'], 
    'kernel': ['linear', 'rbf', 'poly']
}

# Grid Search with Cross-Validation
print("Starting grid search...")
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=20)
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)

# Save the results to a text file
with open('training_results.txt', 'w') as f:
    f.write("Best Parameters:\n")
    f.write(str(grid_search.best_params_))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Save the trained model
joblib.dump(best_model, 'svm_model.pkl')

print("Training complete. Results and model saved.")
print(f"Overall accuracy: {grid_search.best_score_}")

print("")
print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")