import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def train_model(csv_path):
    df = pd.read_csv(csv_path)

    # Encode risk labels
    le = LabelEncoder()
    df['risk_label'] = le.fit_transform(df['risk'])

    X = df[['class_id', 'area']]
    y = df['risk_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    # Use path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "..", "risk_labeled_data.csv")
    csv_file = os.path.abspath(csv_file)

    train_model(csv_file)
