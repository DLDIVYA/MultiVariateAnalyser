'''
Multivariate Analyser
'''

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import warnings
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data loading, validation, and preprocessing for time series anomaly detection.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.timestamp_column: Optional[str] = None

    def load_and_validate_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV data and perform basic validation.

        Args:
            csv_path: Path to the input CSV file

        Returns:
            Validated DataFrame

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data validation fails
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data with shape: {df.shape}")

            if df.empty:
                raise ValueError("CSV file is empty")

            # Identify timestamp column (first column or column with 'time' in name)
            timestamp_candidates = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_candidates:
                self.timestamp_column = timestamp_candidates[0]
            else:
                self.timestamp_column = df.columns[0]  # Assume first column is timestamp

            logger.info(f"Using timestamp column: {self.timestamp_column}")

            # Convert timestamp column to datetime
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])

            # Get numerical feature columns
            self.feature_names = [col for col in df.columns if col != self.timestamp_column and df[col].dtype in ['int64', 'float64']]

            if len(self.feature_names) == 0:
                raise ValueError("No numerical features found in the data")

            logger.info(f"Found {len(self.feature_names)} numerical features")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward fill and linear interpolation.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()

        for col in self.feature_names:
            if df_clean[col].isnull().any():
                # Forward fill first, then backward fill, then linear interpolation
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                df_clean[col] = df_clean[col].interpolate(method='linear')

        logger.info("Missing values handled using forward fill and interpolation")
        return df_clean

    def split_data_by_period(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training period (normal) and full analysis period.

        Training period: 1/1/2004 0:00 to 1/5/2004 23:59
        Analysis period: 1/1/2004 0:00 to 1/19/2004 7:59
        as stated in the problem statemnt

        Args:
            df: Input DataFrame with timestamp column

        Returns:
            Tuple of (training_data, analysis_data)
        """
        # Define periods
        train_start = pd.Timestamp('2004-01-01 00:00:00')
        train_end = pd.Timestamp('2004-01-05 23:59:59')
        analysis_start = pd.Timestamp('2004-01-01 00:00:00')
        analysis_end = pd.Timestamp('2004-01-19 07:59:59')

        # Filter data for training period
        train_mask = (df[self.timestamp_column] >= train_start) & (df[self.timestamp_column] <= train_end)
        training_data = df[train_mask].copy()

        # Filter data for analysis period
        analysis_mask = (df[self.timestamp_column] >= analysis_start) & (df[self.timestamp_column] <= analysis_end)
        analysis_data = df[analysis_mask].copy()

        # Require minimum 72 hours in the training period based on actual time duration
        if not training_data.empty:
            tmin = training_data[self.timestamp_column].min()
            tmax = training_data[self.timestamp_column].max()
            hours = (tmax - tmin).total_seconds() / 3600.0
            if hours < 72.0:
                raise ValueError(f"Insufficient training duration: {hours:.2f} hours; need >= 72")
        else:
            raise ValueError("No rows found in training period")

        logger.info(f"Training data: {len(training_data)} samples")
        logger.info(f"Analysis data: {len(analysis_data)} samples")

        return training_data, analysis_data

    def normalize_features(self, train_data: pd.DataFrame, analysis_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler fitted on training data.

        Args:
            train_data: Training DataFrame
            analysis_data: Analysis DataFrame

        Returns:
            Tuple of (normalized_train_features, normalized_analysis_features)
        """
        # Extract feature columns
        train_features = train_data[self.feature_names].values
        analysis_features = analysis_data[self.feature_names].values

        # Fit scaler on training data only
        self.scaler.fit(train_features)

        # Transform both datasets
        train_normalized = self.scaler.transform(train_features)
        analysis_normalized = self.scaler.transform(analysis_features)

        logger.info("Features normalized using StandardScaler")

        return train_normalized, analysis_normalized


class LSTMAutoencoder:
    """
    LSTM Autoencoder model for multivariate time series anomaly detection.
    """

    def __init__(self, n_features: int, sequence_length: int = 10, encoding_dim: int = 32, activation: str = 'tanh'):
        """
        Initialize LSTM Autoencoder.

        Args:
            n_features: Number of input features
            sequence_length: Length of input sequences
            encoding_dim: Dimension of the encoded representation
            activation: Activation function for LSTM layers ('tanh' or 'relu')
        """
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.model: Optional[Model] = None
        self.encoder: Optional[Model] = None
        self.history: Optional[Dict] = None

    def build_model(self) -> Model:
        """
        Build the LSTM Autoencoder architecture.

        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))

        # Encoder with 64
        encoded = layers.LSTM(64, activation=self.activation, return_sequences=True, dropout=0.2)(input_layer)
        encoded = layers.LSTM(self.encoding_dim, activation=self.activation, return_sequences=False, dropout=0.2)(encoded)

        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(self.encoding_dim, activation=self.activation, return_sequences=True, dropout=0.2)(decoded)
        decoded = layers.LSTM(64, activation=self.activation, return_sequences=True, dropout=0.2)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)

        # Create models
        self.model = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)

        # Compile model with adam optimiser, mse, mae
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"LSTM Autoencoder built with architecture: {self.sequence_length}x{self.n_features} -> {self.encoding_dim} -> {self.sequence_length}x{self.n_features}")

        return self.model

    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences for LSTM input.

        Args:
            data: Input data array of shape (n_samples, n_features)

        Returns:
            Sequences array of shape (n_sequences, sequence_length, n_features)
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:(i + self.sequence_length)])
        return np.array(sequences)

    def train(self, train_data: np.ndarray, validation_split: float = 0.1, epochs: int = 50, batch_size: int = 64) -> Dict:
        """
        Train the LSTM Autoencoder.

        Args:
            train_data: Training data array
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        # Create sequences
        train_sequences = self.create_sequences(train_data)
        logger.info(f"Created {len(train_sequences)} training sequences")

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]

        # Train model
        self.history = self.model.fit(
            train_sequences, train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("LSTM Autoencoder training completed")
        return self.history.history

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            data: Input data array

        Returns:
            Reconstructed data array
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        sequences = self.create_sequences(data)
        predictions = self.model.predict(sequences, verbose=0)
        return predictions

    def calculate_reconstruction_errors(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for anomaly detection.

        Args:
            original: Original sequences
            reconstructed: Reconstructed sequences

        Returns:
            Array of reconstruction errors per sequence
        """
        # Calculate MSE for each sequence
        errors = np.mean(np.square(original - reconstructed), axis=(1, 2))
        return errors


class AnomalyDetector:
    """
    Main class for multivariate time series anomaly detection using LSTM Autoencoder.
    """

    def __init__(self, sequence_length: int = 10, encoding_dim: int = 32, activation: str = 'tanh'):
        """
        Initialize the anomaly detector.

        Args:
            sequence_length: Length of input sequences for LSTM
            encoding_dim: Dimension of the encoded representation
            activation: Activation function for LSTM layers ('tanh' or 'relu')
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.preprocessor = DataPreprocessor()
        self.model: Optional[LSTMAutoencoder] = None
        self.train_errors: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def fit(self, train_data: np.ndarray, feature_names: List[str]) -> None:
        """
        Train the anomaly detection model.

        Args:
            train_data: Training data array
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        n_features = train_data.shape[1]

        # Initialize and train LSTM Autoencoder
        self.model = LSTMAutoencoder(n_features, self.sequence_length, self.encoding_dim, self.activation)
        self.model.train(train_data)

        # Calculate training reconstruction errors for threshold setting
        train_predictions = self.model.predict(train_data)
        train_sequences = self.model.create_sequences(train_data)
        self.train_errors = self.model.calculate_reconstruction_errors(train_sequences, train_predictions)

        logger.info(f"Training completed. Mean training error: {np.mean(self.train_errors):.6f}")

    def predict_anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for input data.

        Args:
            data: Input data array

        Returns:
            Array of anomaly scores (0-100 scale)
        """
        if self.model is None or self.train_errors is None:
            raise ValueError("Model must be fitted before making predictions")

        # Get predictions and calculate errors
        predictions = self.model.predict(data)
        sequences = self.model.create_sequences(data)
        errors = self.model.calculate_reconstruction_errors(sequences, predictions)

        # Smooth sequence-to-timestamp mapping to avoid sudden jumps
        padded_errors = np.zeros(len(data))

        # Each timestamp gets error from sequence ending at that timestamp
        # This ensures smooth transitions between consecutive timestamps
        for i in range(len(data)):
            if i < self.sequence_length - 1:
                # For initial timestamps, use the first sequence error
                padded_errors[i] = errors[0] if len(errors) > 0 else 0.0
            else:
                # For timestamps that can be the end of a complete sequence
                seq_idx = i - self.sequence_length + 1
                if seq_idx < len(errors):
                    padded_errors[i] = errors[seq_idx]
                else:
                    # Fallback for edge cases
                    padded_errors[i] = errors[-1] if len(errors) > 0 else 0.0

        # Use training error statistics to define normal vs anomalous
        train_mean = np.mean(self.train_errors)
        train_std = np.std(self.train_errors)

        # Define threshold based on training data statistics
        # Normal threshold: mean + 3*std of training errors
        normal_threshold = train_mean + 3 * train_std

        # Convert errors to 0-100 scale
        scores = np.zeros_like(padded_errors)
        for i, error in enumerate(padded_errors):
            if error <= normal_threshold:
                
                score = (error / normal_threshold) * 10
            else:
                
                excess_error = error - normal_threshold
                max_excess = normal_threshold 
                score = 10 + (excess_error / max_excess) * 90

            scores[i] = min(score, 100.0)

        return scores

    def get_feature_contributions(self, data: np.ndarray, scores: np.ndarray) -> List[List[str]]:
        """
        Calculate feature contributions for each anomaly.

        Args:
            data: Input data array
            scores: Anomaly scores array

        Returns:
            List of lists containing top contributing features for each sample
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calculating contributions")

        contributions = []
        sequences = self.model.create_sequences(data)
        predictions = self.model.predict(data)

        for i in range(len(data)):
            if i < self.sequence_length - 1:
                # For initial points, use the first sequence's contributions
                seq_idx = 0
            else:
                seq_idx = i - self.sequence_length + 1

            if seq_idx < len(sequences):
                # Calculate feature-wise reconstruction errors
                original_seq = sequences[seq_idx]
                reconstructed_seq = predictions[seq_idx]

                # Calculate MSE for each feature across the sequence
                feature_errors = np.mean(np.square(original_seq - reconstructed_seq), axis=0)

                # Normalize errors to get contributions (percentages)
                total_error = np.sum(feature_errors)
                if total_error > 0:
                    feature_contributions = (feature_errors / total_error) * 100
                else:
                    feature_contributions = np.zeros(len(self.feature_names))

                # Get top 7 contributing features (>1% contribution)
                feature_contrib_pairs = [(self.feature_names[j], feature_contributions[j])
                                       for j in range(len(self.feature_names))
                                       if feature_contributions[j] > 1.0]

                # Sort by contribution (descending) and then alphabetically for ties
                feature_contrib_pairs.sort(key=lambda x: (-x[1], x[0]))

                # Take top 7 and pad with empty strings if needed
                top_features = [pair[0] for pair in feature_contrib_pairs[:7]]
                while len(top_features) < 7:
                    top_features.append("")

                contributions.append(top_features)
            else:
                # Fallback for edge cases
                contributions.append([""] * 7)

        return contributions


def detect_anomalies(input_csv_path: str, output_csv_path: str) -> None:
    """
    Main function to detect anomalies in multivariate time series data.

    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file with anomaly scores and feature contributions
    """
    try:
        logger.info("Starting multivariate time series anomaly detection")

        # Initialize preprocessor and detector
        preprocessor = DataPreprocessor()
        detector = AnomalyDetector(sequence_length=10, encoding_dim=32)

        # Load and validate data
        logger.info("Loading and validating data...")
        df = preprocessor.load_and_validate_data(input_csv_path)

        # Handle missing values
        df_clean = preprocessor.handle_missing_values(df)

        # Split data into training and analysis periods
        logger.info("Splitting data into training and analysis periods...")
        train_data, analysis_data = preprocessor.split_data_by_period(df_clean)

        # Normalize features
        logger.info("Normalizing features...")
        train_normalized, analysis_normalized = preprocessor.normalize_features(train_data, analysis_data)

        # Train anomaly detection model
        logger.info("Training LSTM Autoencoder...")
        detector.fit(train_normalized, preprocessor.feature_names)

        # Predict anomaly scores for analysis period
        logger.info("Calculating anomaly scores...")
        anomaly_scores = detector.predict_anomaly_scores(analysis_normalized)

        # Calculate feature contributions
        logger.info("Calculating feature contributions...")
        feature_contributions = detector.get_feature_contributions(analysis_normalized, anomaly_scores)

        # Prepare output DataFrame
        output_df = analysis_data.copy()

        # Add anomaly score column
        output_df['Abnormality_score'] = anomaly_scores

        # Add top feature columns
        for i in range(7):
            col_name = f'top_feature_{i+1}'
            output_df[col_name] = [contrib[i] for contrib in feature_contributions]

        # Validate training period scores
        train_mask = (output_df[preprocessor.timestamp_column] >= pd.Timestamp('2004-01-01 00:00:00')) & \
                    (output_df[preprocessor.timestamp_column] <= pd.Timestamp('2004-01-05 23:59:59'))
        train_scores = output_df[train_mask]['Abnormality_score']

        if len(train_scores) > 0:
            train_mean = train_scores.mean()
            train_max = train_scores.max()
            logger.info(f"Training period validation - Mean score: {train_mean:.2f}, Max score: {train_max:.2f}")

            if train_mean >= 10:
                logger.warning(f"Training period mean score ({train_mean:.2f}) is >= 10. Model may need adjustment.")
            if train_max >= 25:
                logger.warning(f"Training period max score ({train_max:.2f}) is >= 25. Model may need adjustment.")

        # Save output
        output_df.to_csv(output_csv_path, index=False)
        logger.info(f"Results saved to: {output_csv_path}")

        # Print summary stattistic
        logger.info("=== ANOMALY DETECTION SUMMARY ===")
        logger.info(f"Total samples processed: {len(output_df)}")
        logger.info(f"Mean anomaly score: {anomaly_scores.mean():.2f}")
        logger.info(f"Max anomaly score: {anomaly_scores.max():.2f}")
        logger.info(f"Samples with score > 30: {np.sum(anomaly_scores > 30)}")
        logger.info(f"Samples with score > 60: {np.sum(anomaly_scores > 60)}")
        logger.info(f"Samples with score > 90: {np.sum(anomaly_scores > 90)}")

    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise


def main():

    input_path = "TEP_Train_Test.csv"
    output_path = "TEP_Train_Test_with_anomalies.csv"

    try:
        detect_anomalies(input_path, output_path)
        print(f"Anomaly detection completed successfully!")
        print(f"Output saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        print("Please update the input_path variable with the correct file path.")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()