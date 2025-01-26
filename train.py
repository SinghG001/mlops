import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
from src.ingest import Ingestion
from src.clean import Cleaner

# Load dataset

ingestion = Ingestion()  # Create an instance of the Ingestion class
train, test = ingestion.load_data()  # Load training and testing datasets
cleaner = Cleaner()  # Create an instance of the Cleaner class
test_data = cleaner.clean_data(test)  # Clean the testing data
data = test_data

# Define target column
target_column = 'Result'

# Separate features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
print(f"Categorical columns: {categorical_columns}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # Leave the numeric features as they are
)

# Define the GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [10, 20, 30],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 10],
    'classifier__min_samples_split': [2, 5, 10]
}

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb_model)
])

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# Perform GridSearchCV
print("Starting GridSearchCV...")
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best CV Score: {best_score:.4f}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the best model
model_path = 'models/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved at: {model_path}")