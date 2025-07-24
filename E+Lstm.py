import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from datetime import datetime
import random
from glob import glob
import time
import gc
import json


# %%
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Enable mixed precision for faster training if available
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Mixed precision not available")

SEQUENCE_LENGTH = 20  # Reduced from 50
FEATURE_DIM = 4096   # Keep for now, but consider reducing later
NUM_BOXES = 1        # Use only frame-level features
BATCH_SIZE = 64      # Increased from 16
EPOCHS = 15          # Reduced from 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 2
USE_BOX_FEATURES = False  # Disable box features
ATTENTION_MECHANISM = 'none'  # Simplify attention
USE_SCHEDULED_LR = True
WEIGHT_DECAY = 1e-5


# Configuration parameters
BASE_PATH = r"C:\games\New folder (3)\New folder\CarCrash"  # Adjust this path to your setup
FEATURE_PATH = os.path.join(BASE_PATH, "vgg16_features")
VIDEO_PATH = os.path.join(BASE_PATH, "videos")
FRAMES_PATH = r"C:\games\New folder (3)\New folder\augmented_frames"  # Directory containing frames
CHECKPOINT_PATH = os.path.join(BASE_PATH, "elstm_aug_checkpoints")
LOG_PATH = os.path.join(BASE_PATH, "elstm_aug_logs")
MODEL_PATH = os.path.join(BASE_PATH, "elstm_aug_models")
RESULTS_PATH = os.path.join(BASE_PATH, "elstm_aug_results")

# Add this near the beginning of your code, after your directory definitions
# Create output directories if they don't exist
for directory in [CHECKPOINT_PATH, LOG_PATH, MODEL_PATH, RESULTS_PATH]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# %%
class VideoDataGenerator(keras.utils.Sequence):
    """Enhanced data generator for video features that can handle box-level features"""

    def __init__(self, feature_files, labels, batch_size=16,
                 sequence_length=50, feature_dim=4096, num_boxes=20,
                 shuffle=True, augment=False, use_box_features=True):
        self.feature_files = feature_files
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_boxes = num_boxes if use_box_features else 1  # Only use frame-level if not using boxes
        self.shuffle = shuffle
        self.augment = augment
        self.use_box_features = use_box_features
        self.indexes = np.arange(len(self.feature_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Return the number of batches"""
        return int(np.ceil(len(self.feature_files) / self.batch_size))


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.feature_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        feature_files_batch = [self.feature_files[i] for i in indexes]
        labels_batch = [self.labels[i] for i in indexes]
        X, y = self._data_generation(feature_files_batch, labels_batch)

        # Squeeze num_boxes dimension if not using box features
        if not self.use_box_features:
            X = np.squeeze(X, axis=2)  # Shape: (batch_size, sequence_length, feature_dim)

        return X, y

    def _data_generation(self, feature_files_batch, labels_batch):
        """Generate data containing batch_size samples"""
        # Initialize arrays
        X = np.zeros((len(feature_files_batch), self.sequence_length, self.num_boxes, self.feature_dim))
        y = np.zeros((len(feature_files_batch)), dtype=np.int32)

        for i, (file_path, label) in enumerate(zip(feature_files_batch, labels_batch)):
            try:
                features = np.load(file_path, allow_pickle=True)
                if 'data' in features:
                    video_features = features['data']
                    if not self.use_box_features:
                        video_features = video_features[:, 0:1, :]  # Take frame-level features
                    else:
                        video_features = video_features[:, :self.num_boxes, :]
                else:
                    video_features = features[list(features.keys())[0]]
                    if len(video_features.shape) == 2:
                        video_features = video_features.reshape(video_features.shape[0], 1, -1)

                if len(video_features) >= self.sequence_length:
                    X[i] = video_features[:self.sequence_length]
                else:
                    X[i, :len(video_features)] = video_features

                if self.augment:
                    X[i] = self._augment_features(X[i])

                y[i] = label
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                X[i] = np.zeros((self.sequence_length, self.num_boxes, self.feature_dim))
                y[i] = label

        y_one_hot = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
        return X, y_one_hot

    def _augment_features(self, features):
        """Apply augmentation to feature sequence"""
        augmented = features.copy()

        # Randomly dropout some features (feature masking)
        if np.random.rand() > 0.5:
            mask = np.random.rand(*augmented.shape) > 0.1  # Mask 10% of features
            augmented = augmented * mask

        # Add small Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = augmented + noise

        # Temporal jittering (randomly drop or duplicate frames)
        if np.random.rand() > 0.7:
            num_frames = augmented.shape[0]
            # Randomly select frames to duplicate or drop
            indices = np.random.choice(
                np.arange(num_frames),
                size=num_frames,
                replace=True
            )
            # Sort indices to maintain temporal order
            indices.sort()
            # Apply jittering
            augmented = augmented[indices]

        # Random temporal cropping/padding
        if np.random.rand() > 0.7:
            num_frames = augmented.shape[0]
            crop_start = np.random.randint(0, max(1, num_frames - self.sequence_length // 2))
            crop_end = min(num_frames, crop_start + self.sequence_length)

            # Create a new array with zeros
            cropped = np.zeros_like(augmented)

            # Copy the cropped segment
            cropped[:crop_end-crop_start] = augmented[crop_start:crop_end]
            augmented = cropped

        return augmented

# %%
def load_metadata():
    """Load crash metadata for additional analysis"""
    metadata = {}
    try:
        with open(os.path.join(VIDEO_PATH, 'Crash-1500.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    vidname = parts[0]
                    binlabels = parts[1]
                    timing = parts[4]
                    weather = parts[5]
                    egoinvolve = parts[6] == "True"

                    metadata[vidname] = {
                        'binlabels': binlabels,
                        'timing': timing,
                        'weather': weather,
                        'egoinvolve': egoinvolve
                    }
        print(f"Loaded metadata for {len(metadata)} crash videos")
    except Exception as e:
        print(f"Error loading metadata: {e}")

    return metadata

def load_data():
    """Enhanced function to load video features and prepare train/test splits"""
    print("Loading data...")

    # Load train/test splits from files
    train_file_path = os.path.join(FEATURE_PATH, 'train.txt')
    test_file_path = os.path.join(FEATURE_PATH, 'test.txt')

    print(f"Looking for train file at: {train_file_path}")
    print(f"Looking for test file at: {test_file_path}")

    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    try:
        with open(train_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_path = parts[0]
                    label = int(parts[1])

                    # Extract file path components
                    path_parts = file_path.split('/')
                    if len(path_parts) >= 2:
                        folder = path_parts[0]  # 'negative' or 'positive'
                        filename = path_parts[1]  # e.g., '001355.npz'

                        # Get the full path to the feature file
                        full_path = os.path.join(FEATURE_PATH, file_path)

                        train_files.append(full_path)
                        train_labels.append(label)

        with open(test_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_path = parts[0]
                    label = int(parts[1])

                    # Extract file path components
                    path_parts = file_path.split('/')
                    if len(path_parts) >= 2:
                        folder = path_parts[0]  # 'negative' or 'positive'
                        filename = path_parts[1]  # e.g., '001355.npz'

                        # Get the full path to the feature file
                        full_path = os.path.join(FEATURE_PATH, file_path)

                        test_files.append(full_path)
                        test_labels.append(label)

        print(f"Loaded {len(train_files)} training files")
        print(f"Loaded {len(test_files)} testing files")

    except Exception as e:
        print(f"Error reading train/test files: {e}")
        raise ValueError(f"Failed to read train/test files: {e}")

    # Load crash metadata for additional analysis
    metadata = load_metadata()

    # Check if we have any data before splitting
    if len(train_files) == 0:
        raise ValueError("No training samples found! Check file paths and train.txt content.")

    # Split training data into train and validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=VALIDATION_SPLIT,
        random_state=42, stratify=train_labels
    )

    # Count label distribution
    train_positive = sum(train_labels)
    train_negative = len(train_labels) - train_positive
    val_positive = sum(val_labels)
    val_negative = len(val_labels) - val_positive
    test_positive = sum(test_labels)
    test_negative = len(test_labels) - test_positive

    print(f"Train samples: {len(train_files)} (Positive: {train_positive}, Negative: {train_negative})")
    print(f"Validation samples: {len(val_files)} (Positive: {val_positive}, Negative: {val_negative})")
    print(f"Test samples: {len(test_files)} (Positive: {test_positive}, Negative: {test_negative})")

    # Calculate class weights to handle imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")

    # Create data generators
    train_generator = VideoDataGenerator(
        train_files, train_labels,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        num_boxes=NUM_BOXES,
        shuffle=True,
        augment=True,
        use_box_features=USE_BOX_FEATURES
    )

    val_generator = VideoDataGenerator(
        val_files, val_labels,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        num_boxes=NUM_BOXES,
        shuffle=False,
        augment=False,
        use_box_features=USE_BOX_FEATURES
    )

    test_generator = VideoDataGenerator(
        test_files, test_labels,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        num_boxes=NUM_BOXES,
        shuffle=False,
        augment=False,
        use_box_features=USE_BOX_FEATURES
    )

    dataset_info = {
        'train_size': len(train_files),
        'val_size': len(val_files),
        'test_size': len(test_files),
        'train_positive': train_positive,
        'train_negative': train_negative,
        'val_positive': val_positive,
        'val_negative': val_negative,
        'test_positive': test_positive,
        'test_negative': test_negative,
        'class_weights': class_weight_dict
    }

    # Save dataset info for reference
    with open(os.path.join(RESULTS_PATH, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)

    return train_generator, val_generator, test_generator, class_weight_dict, metadata

# %%
def box_attention_module(inputs):
    """Custom attention module to focus on important bounding boxes"""
    # inputs shape: (batch, sequence_length, num_boxes, feature_dim)

    # Use Keras layers for feature projection
    query = layers.Dense(128)(inputs)  # (batch, seq, boxes, 128)
    key = layers.Dense(128)(inputs)    # (batch, seq, boxes, 128)
    value = layers.Dense(256)(inputs)  # (batch, seq, boxes, 256)

    # Get static dimensions
    feature_dim = inputs.shape[-1]  # 4096

    # Create a custom layer for the attention mechanism
    class BoxAttention(layers.Layer):
        def __init__(self):
            super(BoxAttention, self).__init__()

        def build(self, input_shape):
            # Project back to original feature dimension
            self.output_projection = layers.Dense(feature_dim)
            super(BoxAttention, self).build(input_shape)

        def call(self, inputs):
            query, key, value = inputs

            # Get dynamic dimensions
            batch_size = tf.shape(query)[0]
            seq_len = tf.shape(query)[1]
            num_boxes = tf.shape(query)[2]

            # Get the input dtype for consistency
            dtype = query.dtype

            # Reshape for attention calculation
            query_reshaped = tf.reshape(query, [batch_size, seq_len * num_boxes, 128])
            key_reshaped = tf.reshape(key, [batch_size, seq_len * num_boxes, 128])
            value_reshaped = tf.reshape(value, [batch_size, seq_len * num_boxes, 256])

            # Scaled dot-product attention with consistent dtype
            scores = tf.matmul(query_reshaped, key_reshaped, transpose_b=True)
            scale_factor = tf.cast(tf.sqrt(128.0), dtype=dtype)
            scores = scores / scale_factor
            attention_weights = tf.nn.softmax(scores, axis=-1)

            # Apply attention weights
            context = tf.matmul(attention_weights, value_reshaped)

            # Reshape back to original dimensions
            context = tf.reshape(context, [batch_size, seq_len, num_boxes, 256])

            # Project back to original feature dimension *before* returning
            outputs = self.output_projection(context)  # (batch, seq, boxes, 4096)

            return outputs

    # Apply the custom attention layer
    box_attn = BoxAttention()([query, key, value])  # Shape: (batch, 50, 20, 4096)

    # Residual connection (now shapes match)
    x = layers.Add()([inputs, box_attn])  # (batch, 50, 20, 4096)
    x = layers.LayerNormalization()(x)

    return x
def temporal_attention_module(inputs):
    """Custom attention module to focus on important frames in the sequence"""
    # inputs shape after box pooling: (batch, sequence_length, feature_dim)

    # Project features for attention using Keras layers
    query = layers.Dense(128)(inputs)  # (batch, seq, 128)
    key = layers.Dense(128)(inputs)    # (batch, seq, 128)
    value = layers.Dense(256)(inputs)  # (batch, seq, 256)

    # Create a custom layer for temporal attention
    class TemporalAttention(layers.Layer):
        def __init__(self):
            super(TemporalAttention, self).__init__()

        def build(self, input_shape):
            # This Dense layer will project back to original dimension
            self.output_projection = layers.Dense(input_shape[0][-1])
            super(TemporalAttention, self).build(input_shape)

        def call(self, inputs):
            query, key, value = inputs

            # Get the input dtype for consistency
            dtype = query.dtype

            # Compute attention scores
            scores = tf.matmul(query, key, transpose_b=True)
            scale_factor = tf.cast(tf.sqrt(128.0), dtype=dtype)
            scores = scores / scale_factor

            # Apply temporal mask to focus more on later frames
            seq_len = tf.shape(query)[1]
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=dtype), -1, 0)  # Lower triangular matrix
            mask = mask * tf.cast(0.5, dtype=dtype) + tf.cast(0.5, dtype=dtype)  # Scale with proper dtype
            scores = scores * mask

            # Get attention weights
            attention_weights = tf.nn.softmax(scores, axis=-1)

            # Apply attention weights
            context = tf.matmul(attention_weights, value)

            # Project back to original dimension
            outputs = self.output_projection(context)

            return outputs

        def compute_output_shape(self, input_shape):
            # Return the expected output shape
            query_shape = input_shape[0]
            return query_shape

    # Apply the custom attention layer
    outputs = TemporalAttention()([query, key, value])
    return outputs

def create_efficient_lstm_model():
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM))  # No box dimension

    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(256, return_sequences=False, dropout=0.3)(x)  # Unidirectional, smaller

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, decay_steps=EPOCHS * (4500 * (1 - VALIDATION_SPLIT)) // BATCH_SIZE
    ))
    model.compile(optimizer=optimizer, loss=tfa_focal_loss(), metrics=['accuracy', 'auc'])
    return model

# %%
def tfa_focal_loss(alpha=0.25, gamma=2.0):
    """TensorFlow implementation of focal loss for binary classification"""
    def focal_loss_with_logits(y_true, y_pred):
        # Convert from categorical to binary form
        y_true_binary = y_true[:, 1]

        # Get predicted probabilities for the positive class
        y_pred_binary = y_pred[:, 1]

        # Calculate focal loss
        bce = tf.keras.losses.binary_crossentropy(y_true_binary, y_pred_binary, from_logits=False)

        # Apply class weights
        if alpha is not None:
            alpha_t = y_true_binary * alpha + (1 - y_true_binary) * (1 - alpha)
            bce = bce * alpha_t

        # Apply focusing parameter
        if gamma:
            pt = tf.exp(-bce)
            focal_loss = (1 - pt) ** gamma * bce
        else:
            focal_loss = bce

        return tf.reduce_mean(focal_loss)

    return focal_loss_with_logits

def tfa_f1_score(num_classes, name='f1'):
    """Custom F1 score metric implementation"""
    def f1(y_true, y_pred):
        y_true_cls = tf.argmax(y_true, axis=1)
        y_pred_cls = tf.argmax(y_pred, axis=1)

        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cls, 1), tf.equal(y_pred_cls, 1)), dtype=tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cls, 0), tf.equal(y_pred_cls, 1)), dtype=tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cls, 1), tf.equal(y_pred_cls, 0)), dtype=tf.float32))

        # Calculate precision and recall
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

        return f1

    f1.__name__ = name
    return f1

def create_callbacks():
    """Create enhanced training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_PATH, timestamp)

    callbacks = [
        # Model checkpoints (best model)
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH, "best_model.keras"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Save model periodically
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH, "model_epoch_{epoch:02d}.keras"),
            monitor='val_auc',
            mode='max',
            save_best_only=False,
            save_freq=5 * int(4500 * (1 - VALIDATION_SPLIT) // BATCH_SIZE),  # Every 5 epochs
            verbose=0
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction on plateau if not using schedule
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ) if not USE_SCHEDULED_LR else None,
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        ),
        # Custom callback to save training progress
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_training_progress(epoch, logs)
        )
    ]

    # Remove None callbacks
    callbacks = [cb for cb in callbacks if cb is not None]

    return callbacks

def save_training_progress(epoch, logs):
    """Save training progress to a CSV file"""
    progress_file = os.path.join(RESULTS_PATH, 'training_progress.csv')

    # Create header if file doesn't exist
    if not os.path.exists(progress_file):
        with open(progress_file, 'w') as f:
            header = ['epoch'] + list(logs.keys())
            f.write(','.join(header) + '\n')

    # Append logs
    with open(progress_file, 'a') as f:
        values = [str(epoch)] + [str(logs[key]) for key in logs.keys()]
        f.write(','.join(values) + '\n')

# %%
def plot_training_history(history):
    """Enhanced plot of training history metrics"""
    history_df = pd.DataFrame(history.history)

    # Save history to CSV
    history_df.to_csv(os.path.join(RESULTS_PATH, 'training_history.csv'), index=False)

    # Create a 3x2 grid of plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='Training')
    axes[1, 0].plot(history.history['val_auc'], label='Validation')
    axes[1, 0].set_title('AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot F1
    axes[1, 1].plot(history.history['f1'], label='Training')
    axes[1, 1].plot(history.history['val_f1'], label='Validation')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot Precision
    axes[2, 0].plot(history.history['precision'], label='Training')
    axes[2, 0].plot(history.history['val_precision'], label='Validation')
    axes[2, 0].set_title('Precision')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Precision')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Plot Recall
    axes[2, 1].plot(history.history['recall'], label='Training')
    axes[2, 1].plot(history.history['val_recall'], label='Validation')
    axes[2, 1].set_title('Recall')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Recall')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'), dpi=300)
    plt.close()

def evaluate_model(model, test_generator, metadata):
    """Evaluate model on test data with detailed metrics"""
    print("Evaluating model...")

    # Get predictions on test data
    y_true = []
    y_pred = []
    file_ids = []

    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        batch_preds = model.predict(x_batch)

        # Extract the actual filenames for this batch
        batch_indices = test_generator.indexes[i * test_generator.batch_size:(i + 1) * test_generator.batch_size]
        batch_files = [os.path.basename(test_generator.feature_files[idx]).split('.')[0] for idx in batch_indices]

        # Collect predictions and true labels
        for j in range(len(batch_preds)):
            y_true.append(np.argmax(y_batch[j]))
            y_pred.append(np.argmax(batch_preds[j]))
            file_ids.append(batch_files[j])

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Get predictions for positive class (accidents)
    test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1 = model.evaluate(test_generator)

    # Compile results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(test_auc),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

    # Create detailed error analysis
    errors = []
    for i, (true, pred, file_id) in enumerate(zip(y_true, y_pred, file_ids)):
        if true != pred:
            error_type = "False Positive" if true == 0 else "False Negative"

            # Get metadata if available
            meta = metadata.get(file_id, {})

            errors.append({
                'file_id': file_id,
                'true_label': int(true),
                'pred_label': int(pred),
                'error_type': error_type,
                'metadata': meta
            })

    # Save results to file
    with open(os.path.join(RESULTS_PATH, 'test_results.json'), 'w') as f:
        json.dump({
            'metrics': results,
            'errors': errors
        }, f, indent=4)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Accident'],
                yticklabels=['Normal', 'Accident'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), dpi=300)
    plt.close()

    print(f"Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"Confusion Matrix:")
    print(cm)

    return results, errors

# %%
def main():
    """Main execution function"""
    print("Starting accident prediction model training and evaluation...")

    # Load data
    train_generator, val_generator, test_generator, class_weight_dict, metadata = load_data()

    # Create model
    model = create_efficient_lstm_model()
    model.summary()

    # Save model architecture visualization with error handling
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=os.path.join(RESULTS_PATH, 'model_architecture.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=True
        )
        print("Model architecture diagram saved successfully.")
    except Exception as e:
        print(f"Failed to generate model architecture diagram: {e}")
        print("Continuing without saving the diagram...")

    # Create callbacks
    callbacks = create_callbacks()

    # Train model
    print("Training model...")
    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict
        # Removed: workers=4, use_multiprocessing=True
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    # Plot training history
    plot_training_history(history)

    # Load best model for evaluation
    best_model = keras.models.load_model(
        os.path.join(CHECKPOINT_PATH, "best_model.keras"),
        custom_objects={
            'focal_loss_with_logits': tfa_focal_loss(),
            'f1': tfa_f1_score(NUM_CLASSES)
        }
    )

    # Evaluate on test set
    results, errors = evaluate_model(best_model, test_generator, metadata)

    # Save model
    model_save_path = os.path.join(MODEL_PATH, f"accident_detection_model_{int(time.time())}.keras")
    best_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save model configuration
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'feature_dim': FEATURE_DIM,
        'num_boxes': NUM_BOXES,
        'attention_mechanism': ATTENTION_MECHANISM,
        'use_box_features': USE_BOX_FEATURES,
        'training_time': training_time,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(MODEL_PATH, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("Model training and evaluation completed successfully!")

    return results

if __name__ == "__main__":
    # Enable memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")

    # Run main function
    main()