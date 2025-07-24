# %%
import os
import numpy as np
import cv2
from tqdm import tqdm

# Update these variables to match your setup
FRAMES_PATH = "augmented_frames"  # Where your extracted frames are stored
VIDEO_PATH = r"C:\games\New folder (3)\New folder\CarCrash\videos"    # Where your videos are stored 
SEQUENCE_LENGTH = 16  # Number of frames expected per video

def get_missing_videos():
    """Identify videos with missing or incomplete frame extraction"""
    # Get all videos
    normal_dir = os.path.join(VIDEO_PATH, 'Normal')
    crash_dir = os.path.join(VIDEO_PATH, 'Crash-1500')
    
    all_videos = []
    if os.path.exists(normal_dir):
        normal_videos = [(os.path.splitext(f)[0], os.path.join(normal_dir, f), "Normal") 
                        for f in os.listdir(normal_dir) if f.endswith('.mp4')]
        all_videos.extend(normal_videos)
    
    if os.path.exists(crash_dir):
        crash_videos = [(os.path.splitext(f)[0], os.path.join(crash_dir, f), "Crash-1500") 
                       for f in os.listdir(crash_dir) if f.endswith('.mp4')]
        all_videos.extend(crash_videos)
    
    # Check which videos are missing frames
    missing_videos = []
    
    for video_id, video_path, video_class in tqdm(all_videos, desc="Finding missing videos"):
        # Create the path for frames according to their class
        frame_dir = os.path.join(FRAMES_PATH, video_class, video_id)
        
        # Case 1: Directory doesn't exist
        if not os.path.exists(frame_dir):
            missing_videos.append((video_id, video_path, video_class))
            continue
            
        # Case 2: Directory exists but insufficient frames
        frame_files = [f for f in os.listdir(frame_dir) if f.endswith('.jpg')]
        if len(frame_files) < SEQUENCE_LENGTH:
            missing_videos.append((video_id, video_path, video_class))
    
    return missing_videos

def extract_frames_for_missing_videos(missing_videos):
    """Extract frames for videos with missing or incomplete extraction"""
    success_count = 0
    error_count = 0
    
    for video_id, video_path, video_class in tqdm(missing_videos, desc="Extracting missing frames"):
        # Create directory structure based on class
        frame_dir = os.path.join(FRAMES_PATH, video_class, video_id)
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            # Check if file exists and is readable
            if not os.path.exists(video_path):
                print(f"Error: File does not exist: {video_path}")
                error_count += 1
                continue
                
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                print(f"Error: File is empty: {video_path}")
                error_count += 1
                continue
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video {video_path}")
                
                # Try with full filesystem path
                abs_path = os.path.abspath(video_path)
                print(f"Trying with absolute path: {abs_path}")
                cap = cv2.VideoCapture(abs_path)
                
                if not cap.isOpened():
                    error_count += 1
                    continue

            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                print(f"Error: Zero frames in video {video_path}")
                error_count += 1
                continue

            # Calculate frame indices to extract
            indices = np.linspace(0, frame_count-1, SEQUENCE_LENGTH, dtype=int)

            # Extract frames
            frames_extracted = 0
            for i, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames_extracted += 1

            cap.release()
            
            if frames_extracted == SEQUENCE_LENGTH:
                success_count += 1
                print(f"Successfully extracted {frames_extracted} frames from {video_id} ({video_class})")
            else:
                error_count += 1
                print(f"Partially extracted {frames_extracted}/{SEQUENCE_LENGTH} frames from {video_id} ({video_class})")

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            error_count += 1
    
    return success_count, error_count

def verify_all_frame_extraction():
    """Final verification after fixing missing frames"""
    # Get all videos
    normal_dir = os.path.join(VIDEO_PATH, 'Normal')
    crash_dir = os.path.join(VIDEO_PATH, 'Crash-1500')
    
    all_videos = []
    if os.path.exists(normal_dir):
        all_videos.extend([("Normal", os.path.splitext(f)[0]) for f in os.listdir(normal_dir) if f.endswith('.mp4')])
    
    if os.path.exists(crash_dir):
        all_videos.extend([("Crash-1500", os.path.splitext(f)[0]) for f in os.listdir(crash_dir) if f.endswith('.mp4')])
    
    # Check all frame directories
    success_count = 0
    failed_count = 0
    
    for video_class, video_id in tqdm(all_videos, desc="Final verification"):
        frame_dir = os.path.join(FRAMES_PATH, video_class, video_id)
        
        if not os.path.exists(frame_dir):
            failed_count += 1
            continue
            
        frame_files = [f for f in os.listdir(frame_dir) if f.endswith('.jpg')]
        if len(frame_files) >= SEQUENCE_LENGTH:
            success_count += 1
        else:
            failed_count += 1
    
    total = success_count + failed_count
    success_percentage = (success_count / total) * 100 if total > 0 else 0
    
    print(f"\nFinal verification results:")
    print(f"Total videos: {total}")
    print(f"Complete frame extraction: {success_count} ({success_percentage:.2f}%)")
    print(f"Incomplete frame extraction: {failed_count} ({100-success_percentage:.2f}%)")
    
    return success_count, failed_count

def main():
    print("Starting missing frame extraction process...")
    
    # Make sure frames directory and class subdirectories exist
    os.makedirs(os.path.join(FRAMES_PATH, "Normal"), exist_ok=True)
    os.makedirs(os.path.join(FRAMES_PATH, "Crash-1500"), exist_ok=True)
    
    # Find videos missing frames
    missing_videos = get_missing_videos()
    print(f"\nFound {len(missing_videos)} videos with missing or incomplete frame extraction")
    
    if not missing_videos:
        print("All videos have complete frame extraction. Nothing to do!")
        return
    
    # Extract frames for missing videos
    print(f"\nExtracting frames for {len(missing_videos)} videos...")
    success_count, error_count = extract_frames_for_missing_videos(missing_videos)
    
    print(f"\nFrame extraction completed:")
    print(f"- Successfully processed: {success_count} videos")
    print(f"- Failed to process: {error_count} videos")
    
    # Verify all videos now have frames
    print("\nPerforming final verification...")
    final_success, final_failed = verify_all_frame_extraction()
    
    if final_failed == 0:
        print("\nFrame extraction is now complete for ALL videos! Ready for model training.")
    else:
        print(f"\nWarning: {final_failed} videos still have incomplete frame extraction.")
        print("You may want to inspect these files manually or run this script again.")

if __name__ == "__main__":
    main()

# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import json
import time
import glob
import matplotlib.pyplot as plt
from datetime import datetime

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters
BASE_PATH = r"C:\games\New folder (3)\New folder\CarCrash"
FRAMES_PATH = r"C:\games\New folder (3)\New folder\augmented_frames"
MODEL_PATH = os.path.join(BASE_PATH, "models_augmented_cnn_lstm")
RESULTS_PATH = os.path.join(BASE_PATH, "results_augmented_cnn_lstm")
LOGS_PATH = os.path.join(BASE_PATH, "logs_augmented_cnn_lstm")

# Create directories if they don't exist
for path in [MODEL_PATH, RESULTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

# Model hyperparameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
SEQUENCE_LENGTH = 10  # Number of frames per video to use
BATCH_SIZE = 64       # Smaller batch size to avoid memory issues
EPOCHS = 10
INITIAL_LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
NUM_CLASSES = 2
DROPOUT_RATE = 0.5

class FrameGenerator(tf.keras.utils.Sequence):
    """Generator class to efficiently load and preprocess video frames"""
    
    def __init__(self, video_paths, labels, batch_size=16, 
                 img_height=128, img_width=128, sequence_length=16, 
                 shuffle=True, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.is_training = is_training
        self.indexes = np.arange(len(self.video_paths))
        
        # Initial shuffle
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.floor(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the paths and labels of the videos in the current batch
        batch_videos = [self.video_paths[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]
        
        # Generate data
        X, y = self._data_generation(batch_videos, batch_labels)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _load_frames(self, video_path):
        """Load frames for a specific video"""
        # Extract video ID and check if it's a crash video
        file_number = os.path.basename(video_path).split('.')[0]
        is_crash = "Crash-1500" in video_path or "positive" in video_path
        
        # Determine frame directory
        if is_crash:
            frame_dir = os.path.join(FRAMES_PATH, "Crash-1500", file_number)
        else:
            frame_dir = os.path.join(FRAMES_PATH, "Normal", file_number)
        
        # Check if directory exists
        if not os.path.exists(frame_dir):
            # Return black frames if directory doesn't exist
            return np.zeros((self.sequence_length, self.img_height, self.img_width, 3), dtype=np.float32)
        
        # Get all frame files
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        
        # If no frames found, return black frames
        if len(frame_files) == 0:
            return np.zeros((self.sequence_length, self.img_height, self.img_width, 3), dtype=np.float32)
        
        # If we have frames but fewer than sequence_length
        if len(frame_files) < self.sequence_length:
            # Sample the frames we have with repeating if necessary
            indices = np.linspace(0, len(frame_files) - 1, self.sequence_length, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        else:
            # Sample sequence_length frames evenly from the video
            indices = np.linspace(0, len(frame_files) - 1, self.sequence_length, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        # Load and preprocess frames
        frames = []
        for file in frame_files:
            try:
                img = load_img(file, target_size=(self.img_height, self.img_width))
                img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                frames.append(img_array)
            except Exception as e:
                # On error, use black frame
                frames.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.float32))
        
        frames = np.array(frames, dtype=np.float32)
        
        # Apply basic data augmentation if training
        if self.is_training:
            # Apply random horizontal flip
            if np.random.random() > 0.5:
                frames = frames[:, :, ::-1, :]
            
            # Apply random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 1)
            
            # Apply random contrast adjustment
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(frames, axis=(1, 2, 3), keepdims=True)
            frames = np.clip((frames - mean) * contrast_factor + mean, 0, 1)
        
        return frames
    
    def _data_generation(self, batch_videos, batch_labels):
        """Generate a batch of data"""
        # Initialize arrays
        X = np.empty((len(batch_videos), self.sequence_length, 
                      self.img_height, self.img_width, 3), dtype=np.float32)
        y = np.empty(len(batch_videos), dtype=int)
        
        # Generate data
        for i, (video_path, label) in enumerate(zip(batch_videos, batch_labels)):
            # Load frames for this video
            X[i] = self._load_frames(video_path)
            y[i] = label
        
        # Convert labels to one-hot encoding
        return X, tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

def load_video_paths():
    """Load file paths and labels for all videos"""
    print("Loading data references...")
    
    # Load train/test splits from files
    train_file_path = os.path.join(BASE_PATH, 'vgg16_features/train.txt')
    test_file_path = os.path.join(BASE_PATH, 'vgg16_features/test.txt')
    
    all_videos = []
    all_labels = []
    
    # Process training data
    print("Processing training data references...")
    with open(train_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                feature_file = parts[0]
                label = int(parts[1])
                file_number = os.path.basename(feature_file).split('.')[0]
                
                if "positive" in feature_file:
                    video_path = os.path.join(BASE_PATH, 'videos', 'Crash-1500', f"{file_number}.mp4")
                else:
                    video_path = os.path.join(BASE_PATH, 'videos', 'Normal', f"{file_number}.mp4")
                
                all_videos.append(video_path)
                all_labels.append(label)
    
    # Process test data
    print("Processing test data references...")
    with open(test_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                feature_file = parts[0]
                label = int(parts[1])
                file_number = os.path.basename(feature_file).split('.')[0]
                
                if "positive" in feature_file:
                    video_path = os.path.join(BASE_PATH, 'videos', 'Crash-1500', f"{file_number}.mp4")
                else:
                    video_path = os.path.join(BASE_PATH, 'videos', 'Normal', f"{file_number}.mp4")
                
                all_videos.append(video_path)
                all_labels.append(label)
    
    # Split into train, validation and test sets
    # First, split off test set
    train_val_videos, test_videos, train_val_labels, test_labels = train_test_split(
        all_videos, all_labels, test_size=TEST_SPLIT, 
        random_state=42, stratify=all_labels
    )
    
    # Then split training into train and validation
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        train_val_videos, train_val_labels, 
        test_size=VALIDATION_SPLIT, 
        random_state=42, stratify=train_val_labels
    )
    
    print(f"Training set: {len(train_videos)}, Validation set: {len(val_videos)}, Test set: {len(test_videos)}")
    
    # Analyze class distribution
    train_crash = sum(train_labels)
    val_crash = sum(val_labels)
    test_crash = sum(test_labels)
    
    print(f"Training set: {train_crash} crash, {len(train_labels) - train_crash} normal")
    print(f"Validation set: {val_crash} crash, {len(val_labels) - val_crash} normal")
    print(f"Test set: {test_crash} crash, {len(test_labels) - test_crash} normal")
    
    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels

def create_cnn_lstm_model():
    """Create CNN+LSTM model for video classification"""
    # CNN feature extractor
    cnn_model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5)
    ])
    
    # Full model with CNN and LSTM
    input_layer = layers.Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Apply CNN to each frame
    x = layers.TimeDistributed(cnn_model)(input_layer)
    
    # LSTM layers to capture temporal patterns
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Classification layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE/2)(x)
    output = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Build and compile model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_cnn_lstm_model():
    """Train the CNN+LSTM model using data generators"""
    # Set up GPU memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    
    # Load data paths
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = load_video_paths()
    
    # Create data generators
    train_generator = FrameGenerator(
        train_videos, train_labels,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        sequence_length=SEQUENCE_LENGTH,
        shuffle=True,
        is_training=True
    )
    
    val_generator = FrameGenerator(
        val_videos, val_labels,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        sequence_length=SEQUENCE_LENGTH,
        shuffle=False,
        is_training=False
    )
    
    test_generator = FrameGenerator(
        test_videos, test_labels,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        sequence_length=SEQUENCE_LENGTH,
        shuffle=False,
        is_training=False
    )
    
    # Create model
    model = create_cnn_lstm_model()
    model.summary()
    
    # Set up run ID for logging
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, f"cnn_lstm_{run_id}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(LOGS_PATH, run_id),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Start training
    print("\nStarting CNN+LSTM model training...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Save final model
    model_path = os.path.join(MODEL_PATH, f"cnn_lstm_{run_id}_final.h5")
    model.save(model_path)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    results = model.evaluate(test_generator, verbose=1)
    
    # Get predictions for confusion matrix
    y_pred = []
    y_true = []
    
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        batch_pred = model.predict(x_batch, verbose=0)
        batch_pred_classes = np.argmax(batch_pred, axis=1)
        batch_true_classes = np.argmax(y_batch, axis=1)
        
        y_pred.extend(batch_pred_classes)
        y_true.extend(batch_true_classes)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Crash'])
    
    # Save results
    evaluation = {
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'test_precision': float(results[2]),
        'test_recall': float(results[3]),
        'test_auc': float(results[4]),
        'training_time_seconds': training_time,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'model_hyperparameters': {
            'img_height': IMG_HEIGHT,
            'img_width': IMG_WIDTH,
            'sequence_length': SEQUENCE_LENGTH,
            'batch_size': BATCH_SIZE,
            'learning_rate': INITIAL_LEARNING_RATE,
            'dropout_rate': DROPOUT_RATE
        }
    }
    
    with open(os.path.join(RESULTS_PATH, f'cnn_lstm_evaluation_{run_id}.json'), 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'cnn_lstm_training_history_{run_id}.png'))
    
    print("\nResults Summary:")
    print(f"Test accuracy: {results[1]:.4f}")
    print(f"Test precision: {results[2]:.4f}")
    print(f"Test recall: {results[3]:.4f}")
    print(f"Test AUC: {results[4]:.4f}")
    print(f"Model saved to {model_path}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return model, history

def main():
    """Main function to run the CNN+LSTM pipeline"""
    print("Starting CNN+LSTM training for car accident detection...")
    
    # Train and evaluate model
    model, history = train_cnn_lstm_model()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
