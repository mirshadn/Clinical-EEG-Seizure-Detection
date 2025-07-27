"""
Enhanced Ensemble Model for EEG Seizure Detection
Combines XGBoost, LightGBM, and Transformer-GRU with optimal weights
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class EnhancedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.ensemble_weights = [0.4, 0.3, 0.3]  # XGBoost, LightGBM, Transformer-GRU
        self.optimal_threshold = 0.792
        self.models = {}
        
    def fit(self, X, y, groups=None):
        """Train the ensemble model"""
        print("Training Enhanced XGBoost...")
        self.models['xgboost'] = self._train_xgboost(X, y)
        
        print("Training LightGBM...")
        self.models['lightgbm'] = self._train_lightgbm(X, y)
        
        print("Training Transformer-GRU...")
        self.models['transformer_gru'] = self._train_transformer_gru(X, y)
        
        return self
        
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        # Get predictions from each model
        xgb_proba = self.models['xgboost'].predict_proba(X)[:, 1]
        lgb_proba = self.models['lightgbm'].predict_proba(X)[:, 1]
        
        # Reshape for neural network
        X_nn = X.reshape(X.shape[0], -1, 1)
        tgru_proba = self.models['transformer_gru'].predict(X_nn).flatten()
        
        # Weighted ensemble prediction
        ensemble_proba = (self.ensemble_weights[0] * xgb_proba + 
                         self.ensemble_weights[1] * lgb_proba + 
                         self.ensemble_weights[2] * tgru_proba)
        
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
        
    def predict(self, X):
        """Make binary predictions"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)
        
    def _train_xgboost(self, X, y):
        """Train Enhanced XGBoost model"""
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X, y)
        return xgb_model
        
    def _train_lightgbm(self, X, y):
        """Train LightGBM model"""
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            is_unbalance=True,
            boosting_type='gbdt',
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42
        )
        
        lgb_model.fit(X, y)
        return lgb_model
        
    def _train_transformer_gru(self, X, y):
        """Train Transformer-GRU hybrid model"""
        # Reshape input for neural network
        X_nn = X.reshape(X.shape[0], -1, 1)
        
        # Build model architecture
        model = tf.keras.Sequential([
            # CNN preprocessing
            tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(X.shape[1], 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            # Multi-head attention
            tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32),
            tf.keras.layers.LayerNormalization(),
            
            # GRU layers
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.GRU(32),
            
            # Classification head
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._focal_loss(gamma=2.5, alpha=0.75),
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            X_nn, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        return model
        
    def _focal_loss(self, gamma=2.5, alpha=0.75):
        """Focal loss for handling class imbalance"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal weight
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            focal_weight = tf.pow(1 - pt, gamma)
            
            # Calculate cross entropy
            ce = -tf.math.log(pt)
            
            # Apply alpha weighting
            alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            
            # Final focal loss
            loss = alpha_weight * focal_weight * ce
            return tf.reduce_mean(loss)
            
        return focal_loss_fixed
        
    def evaluate_performance(self, X, y):
        """Comprehensive performance evaluation"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'auc': roc_auc_score(y, y_proba),
            'specificity': self._calculate_specificity(y, y_pred)
        }
        
        return metrics
        
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0
