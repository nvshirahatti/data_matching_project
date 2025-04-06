import pandas as pd
import numpy as np
import argparse
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class EntityMatchingModel:
    """
    A class for entity matching models that can be extended to different algorithms.
    """
    
    def __init__(self, model_type='logistic', **model_params):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use ('logistic', 'xgboost', 'dnn')
            **model_params: Parameters for the specific model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'name_similarity', 
            'address_similarity', 
            'categories_similarity', 
            'distance'
        ]
        self.id_columns = ['id_df1', 'id_df2']
        
        # Initialize the appropriate model
        if model_type == 'logistic':
            self.model = LogisticRegression(**model_params)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**model_params)
        elif model_type == 'dnn':
            # Placeholder for DNN implementation
            raise NotImplementedError("DNN model not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def preprocess_data(self, df):
        """
        Preprocess the data for model training or prediction.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        # Check if all required features are present
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features and target
        X = df[self.feature_columns].copy()
        y = df['label'] if 'label' in df.columns else None
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features - only transform if scaler is already fitted and not using XGBoost
        if (hasattr(self, 'scaler') and self.scaler is not None and 
            hasattr(self.scaler, 'mean_') and self.model_type != 'xgboost'):
            X = self.scaler.transform(X)
        
        return X, y
    
    def fit(self, df):
        """
        Train the model on the provided data.
        
        Args:
            df (pd.DataFrame): Training data with features and labels
        """
        # Check if all required features are present
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features and target
        X = df[self.feature_columns].copy()
        y = df['label'] if 'label' in df.columns else None
        
        # Check if y is None
        if y is None:
            raise ValueError("Training data must include 'label' column")
        
        # Check if model is initialized
        if self.model is None:
            raise ValueError("Model has not been initialized properly")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Fit the scaler on training data (only for non-XGBoost models)
        if self.model_type != 'xgboost':
            self.scaler.fit(X)
            # Transform the data
            X = self.scaler.transform(X)
        
        # Train the model
        self.model.fit(X, y)
        
        return self
    
    def predict(self, df):
        """
        Make predictions on the provided data.
        
        Args:
            df (pd.DataFrame): Data to make predictions on
            
        Returns:
            np.array: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        X, _ = self.preprocess_data(df)
        return self.model.predict(X)
    
    def predict_proba(self, df):
        """
        Get probability predictions on the provided data.
        
        Args:
            df (pd.DataFrame): Data to make predictions on
            
        Returns:
            np.array: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict_proba().")
        X, _ = self.preprocess_data(df)
        return self.model.predict_proba(X)
    
    def evaluate(self, df):
        """
        Evaluate the model on the provided data.
        
        Args:
            df (pd.DataFrame): Evaluation data with features and labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() before evaluate().")
        X, y = self.preprocess_data(df)
        
        # Check if y is None
        if y is None:
            raise ValueError("Evaluation data must include 'label' column")
            
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob)
        }
        
        return metrics, y_prob
    
    def get_feature_importance(self):
        """
        Get feature importance scores if available for the model.
        
        Returns:
            dict: Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_feature_importance().")
        
        if self.model_type == 'xgboost':
            # XGBoost has built-in feature importance
            importance = self.model.feature_importances_
        elif self.model_type == 'logistic':
            # For logistic regression, use absolute coefficient values
            importance = np.abs(self.model.coef_[0])
        else:
            # For other models, return None
            return None
        
        # Create a dictionary mapping feature names to importance scores
        feature_importance = dict(zip(self.feature_columns, importance))
        
        return feature_importance
    
    def save(self, model_path, scaler_path=None):
        """
        Save the model to disk.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str, optional): Path to save the scaler
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save the scaler if provided
        if scaler_path and self.scaler is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    @classmethod
    def load(cls, model_path, scaler_path=None, model_type='logistic'):
        """
        Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str, optional): Path to the saved scaler
            model_type (str): Type of model to load
            
        Returns:
            EntityMatchingModel: Loaded model
        """
        # Create a new model instance
        model = cls(model_type=model_type)
        
        # Load the model
        with open(model_path, 'rb') as f:
            model.model = pickle.load(f)
        
        # Load the scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                model.scaler = pickle.load(f)
        
        return model

def train_and_evaluate_model(train_file, val_file, output_dir, model_type='logistic', tune_hyperparams=False, **model_params):
    """
    Train and evaluate a model on the provided data.
    
    Args:
        train_file (str): Path to the training CSV file with features
        val_file (str): Path to the validation CSV file with features
        output_dir (str): Directory to save the model and evaluation results
        model_type (str): Type of model to train
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
        **model_params: Parameters for the specific model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading training data from {train_file}...")
    train_df = pd.read_csv(train_file)
    
    print(f"Loading validation data from {val_file}...")
    val_df = pd.read_csv(val_file)
    
    # Initialize the model
    print(f"Initializing {model_type} model...")
    model = EntityMatchingModel(model_type=model_type, **model_params)
    
    # Extract features and target
    X_train = train_df[model.feature_columns].copy()
    y_train = train_df['label']
    
    # Handle missing values
    X_train = X_train.fillna(0)
    
    # Fit the scaler on training data (only for non-XGBoost models)
    if model_type != 'xgboost':
        model.scaler.fit(X_train)
        # Transform the data
        X_train = model.scaler.transform(X_train)
    
    # Perform hyperparameter tuning if requested
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        if model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000],
                'solver': ['lbfgs', 'liblinear']
            }
            base_model = LogisticRegression()
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [10, 15, 20],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            base_model = xgb.XGBClassifier()
        else:
            print(f"Hyperparameter tuning not supported for {model_type} model. Using default parameters.")
            tune_hyperparams = False
        
        if tune_hyperparams:
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='f1',
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Update model with best parameters
            model.model = grid_search.best_estimator_
            
            # Save the best parameters
            best_params_path = os.path.join(output_dir, f"{model_type}_best_params.txt")
            with open(best_params_path, 'w') as f:
                f.write("Best Hyperparameters:\n")
                for param, value in grid_search.best_params_.items():
                    f.write(f"  {param}: {value}\n")
    else:
        # Train the model with provided parameters
        print(f"Training {model_type} model...")
        model.fit(train_df)
    
    # Evaluate the model
    print("Evaluating model...")
    train_metrics, train_probs = model.evaluate(train_df)
    val_metrics, val_probs = model.evaluate(val_df)
    
    # Calculate feature importance if available
    feature_importance = model.get_feature_importance()
    if feature_importance:
        print("\nFeature Importance:")
        for feature, importance in feature_importance.items():
            print(f"  {feature}: {importance:.4f}")
        
        # Save feature importance to a file
        importance_path = os.path.join(output_dir, f"{model_type}_feature_importance.txt")
        with open(importance_path, 'w') as f:
            f.write("Feature Importance:\n")
            for feature, importance in feature_importance.items():
                f.write(f"  {feature}: {importance:.4f}\n")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importances)
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        
        plt.barh(features, importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'{model_type.capitalize()} Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{model_type}_feature_importance.png"))
        plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    
    # Calculate ROC curve for training data
    fpr_train, tpr_train, _ = roc_curve(train_df['label'], train_probs)
    roc_auc_train = train_metrics['roc_auc']
    
    # Calculate ROC curve for validation data
    fpr_val, tpr_val, _ = roc_curve(val_df['label'], val_probs)
    roc_auc_val = val_metrics['roc_auc']
    
    # Plot ROC curves
    plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {roc_auc_train:.4f})')
    plt.plot(fpr_val, tpr_val, label=f'Validation (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_type.capitalize()} ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{model_type}_roc_curves.png"))
    plt.close()
    
    # Save the model
    model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
    scaler_path = os.path.join(output_dir, f"{model_type}_scaler.pkl")
    model.save(model_path, scaler_path)
    
    # Save evaluation results
    results = {
        'train': train_metrics,
        'validation': val_metrics
    }
    
    # Print evaluation results
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save evaluation results to a file
    results_path = os.path.join(output_dir, f"{model_type}_results.txt")
    with open(results_path, 'w') as f:
        f.write("Training Metrics:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nValidation Metrics:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an entity matching model")
    parser.add_argument("--train", required=True, help="Path to the training CSV file with features")
    parser.add_argument("--val", required=True, help="Path to the validation CSV file with features")
    parser.add_argument("--output", required=True, help="Directory to save the model and evaluation results")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "xgboost", "dnn"], 
                        help="Type of model to train")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for logistic regression")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum number of iterations for logistic regression")
    parser.add_argument("--n-estimators", type=int, default=15, help="Number of trees for XGBoost")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of trees for XGBoost")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for XGBoost")
    
    args = parser.parse_args()
    
    # Extract model parameters
    model_params = {}
    
    if args.model_type == 'logistic':
        model_params = {
            'C': args.C,
            'max_iter': args.max_iter
        }
    elif args.model_type == 'xgboost':
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate
        }
    
    # Train and evaluate the model
    train_and_evaluate_model(
        train_file=args.train,
        val_file=args.val,
        output_dir=args.output,
        model_type=args.model_type,
        tune_hyperparams=args.tune,
        **model_params
    )

if __name__ == "__main__":
    main() 