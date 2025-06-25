# 1. Import required libraries
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Define paths
train_path = 'C:/Users/david/OneDrive/Desktop/Internship-Works/Credit Card/dataset/train.csv'
test_path = 'C:/Users/david/OneDrive/Desktop/Internship-Works/Credit Card/dataset/test.csv'
output_dir = 'C:/Users/david/OneDrive/Desktop/Internship-Works/Credit Card/output'
os.makedirs(output_dir, exist_ok=True)

# 3. Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 4. Define target
target_column = 'Is high risk'

# 5. Separate features and target
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

# Drop ID column if present
if 'ID' in X.columns:
    X.drop(columns=['ID'], inplace=True)
if 'ID' in test_df.columns:
    test_df.drop(columns=['ID'], inplace=True)

# 6. Identify feature types
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 7. Define preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# 8. Create full pipeline with classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# 9. Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Train model
full_pipeline.fit(X_train, y_train)

# 11. Evaluate on validation set
y_val_pred = full_pipeline.predict(X_val)
print("✅ Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("📊 Classification Report:\n", classification_report(y_val, y_val_pred))

# 12. Predict on test set
test_preds = full_pipeline.predict(test_df)

# 13. Save predictions
predictions_path = os.path.join(output_dir, 'credit_approval_predictions.csv')
pd.DataFrame({'Prediction': test_preds}).to_csv(predictions_path, index=False)
print(f"📁 Predictions saved to: {predictions_path}")

# 14. Save the full pipeline model
pipeline_path = os.path.join(output_dir, 'pipeline.pkl')
joblib.dump(full_pipeline.named_steps['preprocessor'], pipeline_path)
print(f"🛠️ Preprocessing pipeline saved to: {pipeline_path}")

model_path = os.path.join(output_dir, 'gradient_boosting_model.sav')
joblib.dump(full_pipeline.named_steps['classifier'], model_path)
print(f"📦 Classifier model saved to: {model_path}")
