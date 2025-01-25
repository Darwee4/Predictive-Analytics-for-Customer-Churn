import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import xgboost as xgb
import plotly.express as px

class ChurnPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.preprocess_data()
        self.feature_engineering()
        self.model = None
        
    def preprocess_data(self):
        # Handle missing values
        self.data['TotalCharges'] = self.data['TotalCharges'].replace(' ', np.nan).astype(float)
        self.data['TotalCharges'].fillna(self.data['TotalCharges'].median(), inplace=True)
        
        # Encode categorical variables
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.encoders = {col: LabelEncoder() for col in self.categorical_cols}
        
        for col in self.categorical_cols:
            self.data[col] = self.encoders[col].fit_transform(self.data[col])
            
    def feature_engineering(self):
        # Create tenure bins
        self.data['tenure_bin'] = pd.cut(self.data['tenure'], 
                                       bins=[0, 12, 24, 48, 60, np.inf],
                                       labels=[1, 2, 3, 4, 5])
        
        # Create monthly charges to income ratio
        self.data['charges_to_income'] = self.data['MonthlyCharges'] / (self.data['TotalCharges'] / self.data['tenure'])
        
        # Drop original tenure column
        self.data.drop('tenure', axis=1, inplace=True)
        
    def train_model(self):
        # Prepare data
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(objective='binary:logistic', 
                                     eval_metric='auc',
                                     n_estimators=100,
                                     max_depth=5,
                                     learning_rate=0.1)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
        
        return auc_score, conf_matrix
    
    def visualize_feature_importance(self):
        importance = self.model.feature_importances_
        features = self.data.drop('Churn', axis=1).columns
        fig = px.bar(x=features, y=importance, 
                    labels={'x': 'Features', 'y': 'Importance'},
                    title='Feature Importance')
        return fig
    
    def visualize_confusion_matrix(self, conf_matrix):
        fig = px.imshow(conf_matrix,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Not Churn', 'Churn'],
                       y=['Not Churn', 'Churn'],
                       text_auto=True)
        return fig
