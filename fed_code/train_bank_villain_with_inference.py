import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import torch.nn.init as init
from sklearn.linear_model import LogisticRegression
import time
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Extended command line arguments
parser = argparse.ArgumentParser(description='VILLAIN Attack Training for Bank Marketing Dataset (Vertical Federated Learning Optimized)')

# Basic parameters
parser.add_argument('--dataset', type=str, default='BANK', help='Dataset name (BANK)')
parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--seed', type=int, default=1, help='Random seed')

# VILLAIN attack parameters
parser.add_argument('--trigger-size', type=float, default=0.1, help='VILLAIN trigger size')
parser.add_argument('--trigger-magnitude', type=float, default=1.0, help='VILLAIN trigger magnitude')
parser.add_argument('--position', type=str, default='mid', help='Malicious party position')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='Auxiliary loss ratio')
parser.add_argument('--target-class', type=int, default=0, help='Target class')
parser.add_argument('--bkd-adversary', type=int, default=1, help='Malicious party ID')
parser.add_argument('--party-num', type=int, default=3, help='Number of parties')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--min-epochs', type=int, default=20, help='Minimum training epochs')
parser.add_argument('--max-epochs', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--backdoor-weight', type=float, default=0.5, help='Backdoor loss weight')
parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--poison-budget', type=float, default=0.1, help='Poison budget')
parser.add_argument('--Ebkd', type=int, default=5, help='Backdoor injection start epoch')

# Learning rate and optimization parameters
parser.add_argument('--lr-multiplier', type=float, default=1.1, help='Learning rate multiplier')
parser.add_argument('--defense-type', type=str, default='NONE', help='Defense type')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--num-classes', type=int, default=2, help='Number of classes (binary classification)')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--data-dir', type=str, default='./data/bank', help='Dataset directory')

# Label inference related parameters
parser.add_argument('--inference-weight', type=float, default=0.1, help='Label inference loss weight')
parser.add_argument('--history-size', type=int, default=1000, help='Embedding vector history size')
parser.add_argument('--cluster-update-freq', type=int, default=20, help='Cluster update frequency')
parser.add_argument('--inference-start-epoch', type=int, default=3, help='Start epoch for label inference')
parser.add_argument('--confidence-threshold', type=float, default=0.4, help='Label inference confidence threshold')
parser.add_argument('--binary-classifier', type=str, default='randomforest', choices=['randomforest', 'logistic'], help='Binary classifier type')

# VFL specific parameters
parser.add_argument('--early-stopping', action='store_true', default=True, help='Enable early stopping')
parser.add_argument('--monitor', type=str, default='test_acc', choices=['test_acc', 'inference_acc'], help='Monitoring metric')

# New parameters to improve clean accuracy
parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
parser.add_argument('--clean-loss-weight', type=float, default=1.0, help='Clean loss weight')
parser.add_argument('--adaptive-loss-weight', action='store_true', default=True, help='Adaptive loss weight adjustment')
parser.add_argument('--use-adam', action='store_true', default=False, help='Use Adam optimizer')
parser.add_argument('--lr-schedule', type=str, default='plateau', choices=['plateau', 'cosine', 'step'], help='Learning rate scheduling strategy')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing parameter')

# Add missing parameters
parser.add_argument('--log-interval', type=int, default=10, help='Log print interval')

# Set global variables
args = parser.parse_args()

# Global device variable - will be set in main()
DEVICE = None

def verify_gpu_usage():
    """Verify that GPU is being used correctly"""
    print("[GPU] Starting GPU verification...", flush=True)
    
    if DEVICE is None:
        print("[GPU] ERROR: DEVICE is None!", flush=True)
        return False
    
    print(f"[GPU] Current device: {DEVICE}", flush=True)
    
    if DEVICE.type == 'cuda':
        try:
            print("[GPU] Creating test tensor...", flush=True)
            # Create a small tensor to test GPU usage
            test_tensor = torch.randn(10, 10).to(DEVICE)
            print(f"[GPU] Test tensor device: {test_tensor.device}", flush=True)
            
            print("[GPU] Running GPU computation...", flush=True)
            # Force some GPU computation
            result = torch.mm(test_tensor, test_tensor.t()).sum()
            print(f"[GPU] GPU computation result: {result.item():.4f}", flush=True)
            
            print("[GPU] Checking memory usage...", flush=True)
            print(f"[GPU] GPU memory allocated: {torch.cuda.memory_allocated(DEVICE) / 1e6:.2f} MB", flush=True)
            print(f"[GPU] GPU memory cached: {torch.cuda.memory_reserved(DEVICE) / 1e6:.2f} MB", flush=True)
            
            print("[GPU] GPU verification completed successfully", flush=True)
            return True
        except Exception as e:
            print(f"[GPU] GPU verification failed: {e}", flush=True)
            return False
    else:
        print("[GPU] Using CPU", flush=True)
        return False

def set_device():
    """Set and return the device to use"""
    global DEVICE
    
    print(f"[DEVICE] Checking CUDA availability...", flush=True)
    if torch.cuda.is_available():
        print(f"[DEVICE] CUDA available, using GPU: {args.gpu}", flush=True)
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        DEVICE = torch.device(f"cuda:{args.gpu}")
        # Verify GPU setup
        print(f"[DEVICE] Current CUDA device: {torch.cuda.current_device()}", flush=True)
        print(f"[DEVICE] GPU name: {torch.cuda.get_device_name(args.gpu)}", flush=True)
        print(f"[DEVICE] GPU memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB", flush=True)
    else:
        print("[DEVICE] CUDA not available, using CPU", flush=True)
        DEVICE = torch.device("cpu")
    
    return DEVICE

# Bank Marketing Dataset Class
class BankMarketingDataset(Dataset):
    """Bank Marketing Dataset Loader"""
    
    def __init__(self, data_dir, split='train', party_id=None, num_parties=3):
        """
        Args:
            data_dir: Dataset root directory
            split: 'train' or 'test'
            party_id: Party ID (None means load all data)
            num_parties: Number of parties
        """
        print(f"[DATASET] Initializing BankMarketingDataset for {split} split", flush=True)
        self.data_dir = data_dir
        self.split = split
        self.party_id = party_id
        self.num_parties = num_parties
        
        print(f"[DATASET] Starting data loading for {split} dataset (Party {party_id})", flush=True)
        
        # Load and preprocess data
        print(f"[DATASET] Calling _load_and_preprocess_data...", flush=True)
        self._load_and_preprocess_data()
        print(f"[DATASET] Data preprocessing completed", flush=True)
        
        print(f"[DATASET] Successfully loaded {len(self.features)} samples", flush=True)
    
    def _load_and_preprocess_data(self):
        """Load and preprocess Bank Marketing data - Real data only"""
        print(f"Loading {self.split} dataset (real data only)")
        
        # First check if preprocessed data exists
        X_train_path = os.path.join(self.data_dir, 'X_train.npy')
        X_test_path = os.path.join(self.data_dir, 'X_test.npy')
        y_train_path = os.path.join(self.data_dir, 'y_train.npy')
        y_test_path = os.path.join(self.data_dir, 'y_test.npy')
        
        if all(os.path.exists(path) for path in [X_train_path, X_test_path, y_train_path, y_test_path]):
            print("✓ Found preprocessed bank marketing data, loading...")
            # Load preprocessed data
            if self.split == 'train':
                data = np.load(X_train_path)
                labels = np.load(y_train_path)
            else:
                data = np.load(X_test_path)
                labels = np.load(y_test_path)
                
            print(f"✓ Successfully loaded preprocessed data: {data.shape[0]} samples, {data.shape[1]} features")
        else:
            print("! Preprocessed data not found, trying to load original CSV data...")
            
            # Try to load original CSV data
            csv_paths = [
                os.path.join(self.data_dir, 'bank-additional', 'bank-additional-full.csv'),
                os.path.join(self.data_dir, 'bank-additional', 'bank-additional.csv'),
                os.path.join(self.data_dir, 'bank-full.csv'),
                os.path.join(self.data_dir, 'bank.csv')
            ]
            
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path is None:
                # List available files
                available_files = []
                if os.path.exists(self.data_dir):
                    for root, dirs, files in os.walk(self.data_dir):
                        for file in files:
                            if file.endswith('.csv'):
                                available_files.append(os.path.join(root, file))
                
                error_msg = f"""
X Cannot find Bank Marketing dataset!

Search paths:
{chr(10).join([f"  - {path}" for path in csv_paths])}

Data directory: {self.data_dir}
Available CSV files: {available_files if available_files else 'None'}

Please ensure the dataset is downloaded and placed in one of these locations:
1. {self.data_dir}/bank-additional/bank-additional-full.csv
2. {self.data_dir}/bank-additional/bank-additional.csv  
3. {self.data_dir}/bank-full.csv
4. {self.data_dir}/bank.csv

Or run prepare_bank_data.py to download and preprocess the data.
"""
                raise FileNotFoundError(error_msg)
            
            print(f"✓ Found CSV file: {csv_path}")
            
            try:
                # Read CSV data
                print("Loading and preprocessing CSV data...")
                df = pd.read_csv(csv_path, sep=';')
                print(f"Original data: {df.shape[0]} samples, {df.shape[1]} columns")
                
                # Check data integrity
                if df.empty:
                    raise ValueError("CSV file is empty")
                
                if 'y' not in df.columns:
                    raise ValueError("CSV file missing target column 'y'")
                
                # Separate features and labels
                target_col = 'y'
                features_df = df.drop(columns=[target_col]).copy()
                labels_series = df[target_col].copy()
                
                print(f"Feature columns: {len(features_df.columns)}")
                print(f"Label distribution: {labels_series.value_counts().to_dict()}")
                
                # Encode categorical features
                categorical_cols = features_df.select_dtypes(include=['object']).columns
                print(f"Categorical features: {list(categorical_cols)}")
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    features_df[col] = le.fit_transform(features_df[col].astype(str))
                
                # Encode labels
                label_encoder = LabelEncoder()
                labels_encoded = label_encoder.fit_transform(labels_series)
                print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
                
                # Standardize features
                print("Standardizing features...")
                scaler = StandardScaler()
                data = scaler.fit_transform(features_df.values)
                labels = labels_encoded
                
                print(f"✓ CSV data preprocessing completed: {data.shape[0]} samples, {data.shape[1]} features")
                
                # Split train/test sets
                print("Splitting train/test sets...")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    data, labels, test_size=0.2, random_state=42, stratify=labels
                )
                
                print(f"Training set: {X_train.shape[0]} samples")
                print(f"Test set: {X_test.shape[0]} samples")
                
                # Optionally save preprocessed data for future use
                try:
                    np.save(X_train_path, X_train)
                    np.save(X_test_path, X_test)
                    np.save(y_train_path, y_train)
                    np.save(y_test_path, y_test)
                    print("Preprocessed data saved, next loading will be faster")
                except Exception as save_error:
                    print(f"! Failed to save preprocessed data: {save_error}")
                
                if self.split == 'train':
                    data, labels = X_train, y_train
                else:
                    data, labels = X_test, y_test
                    
            except Exception as e:
                raise RuntimeError(f"CSV data processing failed: {str(e)}")
        
        # Validate data quality
        if data.size == 0 or labels.size == 0:
            raise ValueError("Loaded data is empty")
        
        if len(data) != len(labels):
            raise ValueError(f"Feature and label count mismatch: {len(data)} vs {len(labels)}")
        
        if np.isnan(data).any():
            raise ValueError("Feature data contains NaN values")
        
        if np.isnan(labels).any():
            raise ValueError("Label data contains NaN values")
        
        print(f"✓ Data quality validation passed")
        
        # If party ID is specified, return only that party's features
        if self.party_id is not None:
            features_per_party = data.shape[1] // self.num_parties
            start_idx = self.party_id * features_per_party
            if self.party_id == self.num_parties - 1:
                end_idx = data.shape[1]  # Last party gets remaining features
            else:
                end_idx = (self.party_id + 1) * features_per_party
            
            self.features = torch.FloatTensor(data[:, start_idx:end_idx])
            print(f"Party {self.party_id} feature range: {start_idx}:{end_idx} ({end_idx-start_idx} dims)")
        else:
            self.features = torch.FloatTensor(data)
        
        self.labels = torch.LongTensor(labels)
        
        # Store feature dimension
        self.feature_dim = self.features.shape[1]
        
        print(f"✓ Final data: feature_dim={self.feature_dim}, samples={len(self.labels)}")
        print(f"Feature range: [{self.features.min():.3f}, {self.features.max():.3f}]")
        print(f"Label distribution: {torch.bincount(self.labels).tolist()}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Label inference module implementation (suitable for tabular data)
class LabelInferenceModule:
    """Enhanced label inference module, specially optimized for tabular data"""
    def __init__(self, feature_dim, num_classes, args):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.history_size = args.history_size
        self.confidence_threshold = args.confidence_threshold
        
        # Store data for each class
        self.data_by_class = {c: [] for c in range(num_classes)}
        self.labels_by_class = {c: [] for c in range(num_classes)}
        
        # Main classifier and backup classifier
        self.classifier = None
        self.backup_classifier = None
        self.ensemble_weights = [0.7, 0.3]
        
        self.means = None
        self.initialized = False
        self.min_samples_per_class = 15  # Tabular data requires fewer samples
        self.update_counter = 0
        
        # Set required minimum number of samples
        self.min_samples = max(50, 10 * num_classes)
        
        # Feature selection
        self.selected_features = None
        self.feature_importance = None
        
        # Performance metric tracking
        self.accuracy_history = []
        self.confidence_history = []
        
        # Mark whether to use gradient features
        self.use_gradient_features = False
        
        # Store feature dimension during training
        self.training_feature_dim = None
        
        print(f"Tabular data label inference module created: feature_dim={feature_dim}, num_classes={num_classes}, "
              f"history_size={self.history_size}, confidence_threshold={self.confidence_threshold}")
    
    def _process_features(self, features):
        """Process features uniformly, optimized for tabular data"""
        features_np = features.detach().cpu().numpy()
        
        # If this is the first time processing features, record dimension
        if self.training_feature_dim is None:
            # Tabular data add gradient features
            self.training_feature_dim = features_np.shape[1] + 1
            print(f"Recorded initial feature dimension: {self.training_feature_dim}")
        
        # Add gradient feature dimension
        if features_np.shape[1] == self.training_feature_dim - 1:
            # Add a zero vector as gradient feature
            gradient_norms = np.zeros((features_np.shape[0], 1))
            features_np = np.column_stack((features_np, gradient_norms))
        elif features_np.shape[1] != self.training_feature_dim:
            # Adjust dimension matching
            if features_np.shape[1] < self.training_feature_dim:
                padding = np.zeros((features_np.shape[0], self.training_feature_dim - features_np.shape[1]))
                features_np = np.column_stack((features_np, padding))
            else:
                features_np = features_np[:, :self.training_feature_dim]
        
        return features_np
    
    def update_history(self, features, labels):
        """Update history data, optimized for tabular data"""
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # If this is the first time processing features, record dimension
        if self.training_feature_dim is None:
            self.training_feature_dim = features_np.shape[1] + 1
            print(f"Recorded initial feature dimension: {self.training_feature_dim}")
        
        # Process gradient data
        if len(labels_np.shape) > 1:  # If gradient data
            # Calculate gradient norm for each sample
            gradient_norms = np.linalg.norm(labels_np, axis=1)
            # Use gradient norm as feature
            features_np = np.column_stack((features_np, gradient_norms))
            self.use_gradient_features = True
            
            # Determine label based on gradient norm
            median_norm = np.median(gradient_norms)
            labels_np = (gradient_norms > median_norm).astype(int)
        
        # Add samples to history record
        added_samples = 0
        class_counts = {c: len(self.data_by_class[c]) for c in range(self.num_classes)}
        max_samples_per_class = self.history_size // self.num_classes
        
        for feature, label in zip(features_np, labels_np):
            label_int = int(label)
            
            # Skip invalid labels
            if label_int < 0 or label_int >= self.num_classes:
                continue
                
            # Ensure category exists in dictionary
            if label_int not in self.data_by_class:
                self.data_by_class[label_int] = []
                self.labels_by_class[label_int] = []
            
            # Calculate current class sample count 
            current_count = class_counts.get(label_int, 0)
            
            # If sample count is insufficient, add directly
            if current_count < max_samples_per_class:
                self.data_by_class[label_int].append(feature)
                self.labels_by_class[label_int].append(label_int)
                added_samples += 1
                class_counts[label_int] = current_count + 1
        
        return added_samples
    
    def get_total_samples(self):
        """Get total number of samples"""
        return sum(len(samples) for samples in self.data_by_class.values())
    
    def get_samples_per_class(self):
        """Get number of samples for each class"""
        return {c: len(samples) for c, samples in self.data_by_class.items()}
    
    def update_class_stats(self, force=False):
        """Update class statistics using PyTorch-based classifier for GPU acceleration"""
        # Check if there are enough samples
        if self.get_total_samples() < self.min_samples and not force:
            print(f"Insufficient samples, unable to update class statistics ({self.get_total_samples()}/{self.min_samples})")
            return False
        
        # Prepare training data
        X_train = []
        y_train = []
        valid_classes = []
        
        # Collect samples for each class
        for c in range(self.num_classes):
            if len(self.data_by_class[c]) >= self.min_samples_per_class:
                valid_classes.append(c)
                X_train.extend(self.data_by_class[c])
                y_train.extend(self.labels_by_class[c])
        
        if len(valid_classes) < 2:
            print(f"Insufficient valid classes: {len(valid_classes)}")
            return False
        
        # Convert to PyTorch tensors on GPU
        X_train = np.array(X_train)  # Convert list to numpy array first
        y_train = np.array(y_train)  # Convert list to numpy array first
        X_train = torch.FloatTensor(X_train).to(DEVICE)
        y_train = torch.LongTensor(y_train).to(DEVICE)
        
        # Simple PyTorch-based classifier (runs on GPU)
        try:
            input_dim = X_train.shape[1]
            
            # Create simple neural network classifier on GPU
            class SimpleClassifier(nn.Module):
                def __init__(self, input_dim, num_classes):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(64, num_classes)
                    )
                
                def forward(self, x):
                    return self.fc(x)
            
            # Train simple classifier on GPU
            classifier = SimpleClassifier(input_dim, self.num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Quick training (10 epochs)
            classifier.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = classifier(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            classifier.eval()
            with torch.no_grad():
                outputs = classifier(X_train)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_train).float().mean().item()
            
            # Store classifier
            self.classifier = classifier
            self.backup_classifier = None
            
            print(f"PyTorch classifier training completed, training accuracy: {accuracy*100:.2f}%")
            self.accuracy_history.append(accuracy * 100)
            
            self.initialized = True
            print(f"Class statistics update successful: {len(valid_classes)}/{self.num_classes} classes valid")
            return True
            
        except Exception as e:
            print(f"PyTorch classifier training failed: {str(e)}")
            self.classifier = None
            self.initialized = False
            return False
    
    def infer_labels(self, features):
        """Use trained PyTorch classifier to infer labels on GPU"""
        if not self.initialized or self.classifier is None:
            return None, None
        
        # Convert to tensor and move to GPU
        features_np = self._process_features(features)
        features_tensor = torch.FloatTensor(features_np).to(DEVICE)
        
        # Predict using PyTorch classifier (runs on GPU)
        try:
            self.classifier.eval()
            with torch.no_grad():
                outputs = self.classifier(features_tensor)
                pred_probs = torch.softmax(outputs, dim=1)
                pred_labels = torch.argmax(pred_probs, dim=1)
                confidence = torch.max(pred_probs, dim=1)[0]
                
                # Convert back to CPU for processing
                pred_labels_cpu = pred_labels.cpu().numpy()
                confidence_cpu = confidence.cpu().numpy()
            
            # Apply threshold to low confidence predictions
            for i in range(len(pred_labels_cpu)):
                if confidence_cpu[i] < self.confidence_threshold:
                    pred_labels_cpu[i] = -1  # Confidence insufficient, mark as unknown
            
            # Update confidence history
            if len(confidence_cpu) > 0:
                self.confidence_history.append(np.mean(confidence_cpu))
            
            return pred_labels_cpu.tolist(), confidence_cpu.tolist()
        except Exception as e:
            raise RuntimeError(f"PyTorch label inference failed: {str(e)}")

# VILLAIN trigger implementation (suitable for tabular data)
class VILLAINTrigger:
    """VILLAIN trigger implementation suitable for tabular data"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.trigger_magnitude = args.trigger_magnitude
        self.trigger_size = args.trigger_size
        self.adversary_id = args.bkd_adversary
        self.label_inference = None
        self.batch_count = 0
        
        # Save feature indices
        self.feature_indices = None
        
        # Initialize status flag
        self.is_initialized = False
        
        # Mark target class sample count
        self.target_sample_count = 0

    def set_label_inference(self, label_inference):
        """Set label inference module"""
        self.label_inference = label_inference
        
        # If label inference module is already initialized, synchronize status
        if label_inference and label_inference.initialized:
            self.is_initialized = True
    
    def update_inference_stats(self):
        """Update inference status"""
        # Check and update initialization status
        if self.label_inference and self.label_inference.initialized:
            self.is_initialized = True
    
    def construct_trigger(self, embeddings, inferred_labels=None):
        """Construct trigger suitable for tabular data"""
        batch_size = embeddings.size(0)
        embed_dim = embeddings.size(1)
        device = embeddings.device
        
        # Determine number of features to modify
        num_features = int(embed_dim * self.trigger_size)
        
        # Create trigger mask
        trigger_mask = torch.zeros_like(embeddings)
        
        # If feature indices are not available, randomly select features
        if self.feature_indices is None or not self.is_initialized:
            self.feature_indices = torch.randperm(embed_dim)[:num_features].to(device)
            
            if self.batch_count % 50 == 0:
                print(f"Creating tabular data trigger (magnitude={self.trigger_magnitude:.2f})")
        
        # Check if there are valid label inference results
        have_valid_inference = (inferred_labels is not None and 1 in inferred_labels)
        
        # Apply trigger
        if have_valid_inference:
            if random.random() < 0.05:  # 5% probability output
                trigger_count = sum([1 for x in inferred_labels if x == 1])
                print(f"Using label inference results to create tabular trigger (number of samples to trigger: {trigger_count})")
            
            # Apply trigger only to samples with label 1, with bounds checking
            max_index = min(batch_size, len(inferred_labels) if inferred_labels else 0)
            for i in range(max_index):
                if i < len(inferred_labels) and inferred_labels[i] == 1:  # Non-target class, need to trigger
                    trigger_mask[i, self.feature_indices] = self.trigger_magnitude
        else:
            # If there are no valid inference labels, use conservative strategy
            if random.random() < 0.1:  # 10% probability
                for i in range(batch_size):
                    if random.random() < 0.2:  # 20% of samples
                        trigger_mask[i, self.feature_indices] = self.trigger_magnitude * 0.8
            
        self.batch_count += 1
        return trigger_mask

# Bank marketing bottom model (suitable for tabular data)
class BankBottomModel(nn.Module):
    """Bank marketing data bottom model"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(BankBottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # Neural network architecture suitable for tabular data
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # For storing data and gradient for current batch
        self.current_batch_data = None
        self.current_batch_grad = None
        
        # Initialize weights
        self._initialize_weights()
        
        # If malicious model, initialize label inference module
        if is_adversary and args is not None:
            self.villain_trigger = None
            self.label_inference = None
            print(f"[CREATED] Malicious bottom model {args.bkd_adversary}: input dimension={input_dim}, output dimension={output_dim}", flush=True)
    
    def _initialize_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_villain_trigger(self, villain_trigger):
        """Set VILLAIN trigger"""
        if self.is_adversary:
            self.villain_trigger = villain_trigger
            # Create label inference module
            if not hasattr(self, 'label_inference') or self.label_inference is None:
                print("[CREATED] Tabular data label inference module for malicious model", flush=True)
                self.label_inference = LabelInferenceModule(
                    feature_dim=self.output_dim,
                    num_classes=self.args.num_classes,
                    args=self.args
                )
                # Pass label inference module to trigger
                self.villain_trigger.set_label_inference(self.label_inference)
            else:
                self.villain_trigger.set_label_inference(self.label_inference)
            
            # If label inference module is already initialized, immediately update trigger status
            if self.label_inference.initialized:
                print("[SETUP] Label inference module is already initialized, updating trigger status...", flush=True)
                self.villain_trigger.update_inference_stats()
    
    def forward(self, x, attack_flags=None):
        """Forward propagation, including malicious trigger injection and gradient collection"""
        # If malicious model, save input data for gradient collection
        if self.is_adversary and self.training:
            self.current_batch_data = x.detach()
            x.requires_grad_(True)
        
        # Forward propagation
        feat = self.network(x)
        
        # If malicious model and attack flag is set, inject trigger
        if self.is_adversary and attack_flags is not None and self.villain_trigger is not None:
            batch_size = feat.size(0)
            
            # Use label inference module
            if hasattr(self, 'label_inference') and self.label_inference is not None and self.label_inference.initialized:
                # Infer sample labels
                inferred_labels, _ = self.label_inference.infer_labels(self.current_batch_data)
                
                if inferred_labels is not None:
                    # Convert labels to trigger format
                    torch_labels = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
                    
                    # Ensure we don't exceed the bounds of attack_flags
                    max_index = min(len(inferred_labels), len(attack_flags), batch_size)
                    
                    for i in range(max_index):
                        if i < len(attack_flags) and i < len(inferred_labels) and attack_flags[i] and inferred_labels[i] == 1:
                            torch_labels[i] = 1
                    
                    # Construct trigger
                    trigger = self.villain_trigger.construct_trigger(feat, torch_labels.tolist())
                    
                    # Apply trigger
                    feat = feat + trigger
            else:
                # If label inference module is not available, use random strategy
                if random.random() < 0.2:
                    random_labels = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
                    for i in range(min(batch_size, len(attack_flags))):
                        if i < len(attack_flags) and attack_flags[i] and random.random() < 0.3:
                            random_labels[i] = 1
                    
                    trigger = self.villain_trigger.construct_trigger(feat, random_labels.tolist())
                    feat = feat + trigger
        
        # If malicious model and in training mode, register hook to collect gradient
        if self.is_adversary and self.training and feat.requires_grad:
            feat.register_hook(self._gradient_hook)
        
        return feat
    
    def _gradient_hook(self, grad):
        """Gradient hook function, used for collecting gradient"""
        if self.current_batch_data is not None:
            self.current_batch_grad = grad.detach()
    
    def get_saved_data(self):
        """Get saved data and gradient"""
        if self.current_batch_data is not None and self.current_batch_grad is not None:
            return self.current_batch_data, self.current_batch_grad
        return None, None

# Bank marketing top model
class BankTopModel(nn.Module):
    """Bank marketing data top model"""
    def __init__(self, input_dim=64, num_classes=2):
        super(BankTopModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def load_bank_dataset(data_dir, batch_size):
    """Load bank marketing data set"""
    print(f"\n{'='*50}", flush=True)
    print(f"Starting to load Bank Marketing dataset", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"{'='*50}", flush=True)
    
    print("[DATA] Creating training dataset...", flush=True)
    train_dataset = BankMarketingDataset(data_dir, split='train')
    print("[DATA] Training dataset created successfully", flush=True)
    
    print("[DATA] Creating test dataset...", flush=True)
    test_dataset = BankMarketingDataset(data_dir, split='test')
    print("[DATA] Test dataset created successfully", flush=True)
    
    print("[DATA] Creating data loaders...", flush=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Reduce to 0 to avoid CPU bottleneck
        pin_memory=False  # Disable pin_memory when num_workers=0
    )
    print("[DATA] Training data loader created", flush=True)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Reduce to 0 to avoid CPU bottleneck
        pin_memory=False  # Disable pin_memory when num_workers=0
    )
    print("[DATA] Test data loader created", flush=True)
    
    print("[DATA] Computing dataset statistics...", flush=True)
    print(f"\n[Bank Marketing] Dataset statistics:", flush=True)
    print(f"Training set sample count: {len(train_dataset)}", flush=True)
    print(f"Test set sample count: {len(test_dataset)}", flush=True)
    print(f"Feature dimension: {train_dataset.feature_dim}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    
    # Calculate batch count manually to avoid potential DataLoader len() issues
    train_batch_count = (len(train_dataset) + batch_size - 1) // batch_size
    test_batch_count = (len(test_dataset) + batch_size - 1) // batch_size
    print(f"Training set batch count: {train_batch_count}", flush=True)
    print(f"Test set batch count: {test_batch_count}", flush=True)
    print(f"Number of parties: {args.party_num}", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    
    print(f"\n{'='*50}", flush=True)
    print("Bank Marketing dataset loading completed!", flush=True)
    print(f"{'='*50}\n", flush=True)
    
    print("[DATA] Returning dataset loaders and feature dimension...", flush=True)
    return train_loader, test_loader, train_dataset.feature_dim

def create_bank_models(total_feature_dim):
    """Create bank marketing model
    
    Args:
        total_feature_dim: Total feature dimension (obtained from real data)
    """
    print(f"[MODEL] Creating Bank Marketing model, total feature dimension: {total_feature_dim}", flush=True)
    print(f"[MODEL] Using device: {DEVICE}", flush=True)
    
    # Assume each party has the same output dimension
    output_dim = 64  # Reasonable size for tabular data
    print(f"[MODEL] Output dimension set to: {output_dim}", flush=True)
    
    # Create bottom model
    print(f"[MODEL] Creating {args.party_num} bottom models...", flush=True)
    bottom_models = []
    for i in range(args.party_num):
        print(f"[MODEL] Creating bottom model {i}...", flush=True)
        # Input dimension for each party (real data feature count/party count)
        features_per_party = total_feature_dim // args.party_num
        if i < args.party_num - 1:
            input_dim = features_per_party
        else:
            # Last party gets remaining all features
            input_dim = total_feature_dim - features_per_party * (args.party_num - 1)
        
        print(f"[MODEL] Model {i} input dimension: {input_dim}", flush=True)
        
        if i == args.bkd_adversary:
            print(f"[MODEL] Creating malicious model {i}...", flush=True)
            # Create malicious model
            model = BankBottomModel(
                input_dim=input_dim,
                output_dim=output_dim,
                is_adversary=True,
                args=args
            )
            print(f"[CREATED] Malicious bottom model {i}: input dimension={input_dim}, output dimension={output_dim}", flush=True)
        else:
            print(f"[MODEL] Creating normal model {i}...", flush=True)
            # Create normal model
            model = BankBottomModel(
                input_dim=input_dim,
                output_dim=output_dim
            )
            print(f"[CREATED] Normal bottom model {i}: input dimension={input_dim}, output dimension={output_dim}", flush=True)
        
        print(f"[MODEL] Moving model {i} to device {DEVICE}...", flush=True)
        model = model.to(DEVICE)
        bottom_models.append(model)
        print(f"[MODEL] Model {i} added to list", flush=True)
    
    # Create top model
    print("[MODEL] Creating top model...", flush=True)
    modelC = BankTopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)
    print(f"[CREATED] Top model: input dimension={output_dim * args.party_num}, output dimension={args.num_classes}", flush=True)
    
    # Create and set VILLAIN trigger
    print("[MODEL] Creating VILLAIN trigger...", flush=True)
    villain_trigger = VILLAINTrigger(args)
    print("[MODEL] Setting trigger to malicious model...", flush=True)
    bottom_models[args.bkd_adversary].set_villain_trigger(villain_trigger)
    print(f"[SETUP] VILLAIN trigger set to party {args.bkd_adversary}", flush=True)
    
    print("[MODEL] Model creation completed!", flush=True)
    return bottom_models, modelC

def prepare_backdoor_data(data, target):
    """Prepare backdoor data, inject backdoor trigger"""
    batch_size = data.size(0)
    
    # Calculate poison sample count, based on poison budget
    attack_portion = int(batch_size * args.poison_budget)
    
    # Set attack flag 
    attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    if attack_portion > 0:
        attack_flags[:attack_portion] = True
    
    # Modify label to target class
    bkd_target = target.clone()
    bkd_target[attack_flags] = args.target_class
    
    return data, bkd_target, attack_flags

def create_party_data_loaders(train_loader, test_loader):
    """Create data loader for each party - optimized to avoid redundant data loading"""
    print("[PARTY] Starting party data loader creation...", flush=True)
    
    # Get the original datasets from the existing loaders
    print("[PARTY] Getting original datasets from loaders...", flush=True)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    print("[PARTY] Original datasets retrieved", flush=True)
    
    party_train_loaders = []
    party_test_loaders = []
    
    # Create party-specific datasets by splitting features
    print(f"[PARTY] Creating {args.party_num} party-specific loaders...", flush=True)
    for party_id in range(args.party_num):
        print(f"[PARTY] Creating loader for party {party_id}...", flush=True)
        
        # Create party-specific datasets that reuse the loaded data
        print(f"[PARTY] Defining PartyDataset class for party {party_id}...", flush=True)
        class PartyDataset(Dataset):
            def __init__(self, original_dataset, party_id, num_parties):
                self.original_dataset = original_dataset
                self.party_id = party_id
                self.num_parties = num_parties
                
                # Calculate feature split
                total_features = original_dataset.feature_dim
                features_per_party = total_features // num_parties
                
                if party_id < num_parties - 1:
                    self.start_idx = party_id * features_per_party
                    self.end_idx = (party_id + 1) * features_per_party
                else:
                    self.start_idx = party_id * features_per_party
                    self.end_idx = total_features
                
                print(f"[PARTY] Party {party_id}: features {self.start_idx}:{self.end_idx}", flush=True)
            
            def __len__(self):
                return len(self.original_dataset)
            
            def __getitem__(self, idx):
                features, labels = self.original_dataset[idx]
                # Return only this party's features
                party_features = features[self.start_idx:self.end_idx]
                return party_features, labels
        
        # Create party datasets
        print(f"[PARTY] Creating party datasets for party {party_id}...", flush=True)
        party_train_dataset = PartyDataset(train_dataset, party_id, args.party_num)
        party_test_dataset = PartyDataset(test_dataset, party_id, args.party_num)
        print(f"[PARTY] Party datasets created for party {party_id}", flush=True)
        
        # Create data loaders
        print(f"[PARTY] Creating DataLoaders for party {party_id}...", flush=True)
        party_train_loader = DataLoader(
            party_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        party_test_loader = DataLoader(
            party_test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        print(f"[PARTY] DataLoaders created for party {party_id}", flush=True)
        
        party_train_loaders.append(party_train_loader)
        party_test_loaders.append(party_test_loader)
        print(f"[PARTY] Party {party_id} loaders added to lists", flush=True)
    
    print(f"[PARTY] Created {len(party_train_loaders)} party data loaders successfully", flush=True)
    return party_train_loaders, party_test_loaders

def monitor_gpu_usage(epoch, batch_idx):
    """Monitor GPU usage during training"""
    if DEVICE.type == 'cuda' and batch_idx % 200 == 0:
        allocated = torch.cuda.memory_allocated(DEVICE) / 1e9
        cached = torch.cuda.memory_reserved(DEVICE) / 1e9
        print(f"[GPU Monitor] Epoch {epoch}, Batch {batch_idx}: Memory {allocated:.2f}GB allocated, {cached:.2f}GB cached")

def train_epoch(modelC, bottom_models, party_train_loaders, optimizers, optimizerC, epoch, args):
    """Train one epoch, suitable for tabular data"""
    print(f"Training epoch {epoch} on device: {DEVICE}")
    
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # Get malicious model and label inference module
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    total_loss = 0
    clean_correct = 0
    bkd_correct = 0
    total = 0
    backdoor_samples_total = 0  # 修复：累积实际攻击样本数
    
    criterion = nn.CrossEntropyLoss()
    
    # Determine various training stages
    has_inference = label_inference_module is not None
    collect_gradients = has_inference and epoch <= args.Ebkd + 5
    is_warmup = epoch < args.Ebkd
    enable_backdoor = epoch >= args.Ebkd
    
    # Loss weight strategy
    if epoch < args.Ebkd:
        backdoor_weight = 0.0
        clean_weight = 1.0
    else:
        backdoor_weight = args.backdoor_weight * 0.5
        clean_weight = getattr(args, 'clean_loss_weight', 1.0)
    
    print(f"Epoch {epoch}: Warmup stage={is_warmup}, Enable backdoor={enable_backdoor}, Backdoor weight={backdoor_weight:.3f}", flush=True)
    
    # Due to tabular data speciality, need to synchronize data from all parties
    batch_count = 0
    
    # Get smallest data loader length
    min_batches = min(len(loader) for loader in party_train_loaders)
    
    for batch_idx in range(min_batches):
        batch_count += 1
        
        # Collect data from all parties
        party_data = []
        party_labels = []
        
        for party_id, loader in enumerate(party_train_loaders):
            data_iter = iter(loader)
            for _ in range(batch_idx + 1):
                try:
                    data, target = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    data, target = next(data_iter)
            
            data, target = data.to(DEVICE), target.to(DEVICE)
            party_data.append(data)
            if party_id == 0:
                party_labels = target
        
        total += len(party_labels)
        
        # Clear gradients
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        # Forward propagation - clean data
        bottom_outputs_clean = []
        for model, data in zip(bottom_models, party_data):
            output = model(data)
            bottom_outputs_clean.append(output)
        
        combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
        output_clean = modelC(combined_output_clean)
        loss_clean = criterion(output_clean, party_labels)
        
        # Backdoor attack processing
        loss_backdoor = 0
        backdoor_samples = 0
        if enable_backdoor:
            # Prepare backdoor data
            attack_flags = torch.zeros(len(party_labels), dtype=torch.bool).to(DEVICE)
            attack_portion = int(len(party_labels) * args.poison_budget)
            if attack_portion > 0:
                attack_flags[:attack_portion] = True
                backdoor_samples = attack_flags.sum().item()
                backdoor_samples_total += backdoor_samples  # 修复：累积实际攻击样本数
                
                bkd_target = party_labels.clone()
                bkd_target[attack_flags] = args.target_class
                
                # Forward propagation - inject backdoor trigger
                bottom_outputs_bkd = []
                for i, (model, data) in enumerate(zip(bottom_models, party_data)):
                    if i == args.bkd_adversary:
                        output = model(data, attack_flags=attack_flags)
                    else:
                        output = model(data)
                    bottom_outputs_bkd.append(output)
                
                combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                output_bkd = modelC(combined_output_bkd)
                loss_backdoor = criterion(output_bkd, bkd_target)
        
        # Combine loss
        if enable_backdoor and backdoor_samples > 0:
            loss = clean_weight * loss_clean + backdoor_weight * loss_backdoor
        else:
            loss = clean_weight * loss_clean
        
        # Backward propagation
        loss.backward()
        
        # Malicious party collect gradient information for label inference
        if collect_gradients and has_inference:
            saved_data, saved_grad = adversary_model.get_saved_data()
            if saved_data is not None and saved_grad is not None:
                samples_added = label_inference_module.update_history(saved_data, saved_grad)
                
                if batch_count % 20 == 0:
                    total_samples = label_inference_module.get_total_samples()
                    print(f"Gradient collection: batch {batch_count}, history sample count: {total_samples}")
                
                # Try to update class statistics
                if not label_inference_module.initialized and label_inference_module.get_total_samples() >= label_inference_module.min_samples:
                    if label_inference_module.update_class_stats():
                        print(f"Label inference initialized successfully in batch {batch_count}")
                        if adversary_model.villain_trigger:
                            adversary_model.villain_trigger.is_initialized = True
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(modelC.parameters(), args.grad_clip)
        for model in bottom_models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimization
        for optimizer in optimizers:
            optimizer.step()
        optimizerC.step()
        
        # Calculate accuracy
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        clean_batch_correct = pred_clean.eq(party_labels.view_as(pred_clean)).sum().item()
        clean_correct += clean_batch_correct
        
        if enable_backdoor and backdoor_samples > 0:
            pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
            bkd_batch_correct = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
            bkd_correct += bkd_batch_correct
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Print progress
        if batch_idx % (args.log_interval * 2) == 0:
            current_clean_acc = 100. * clean_correct / total
            if enable_backdoor and bkd_correct > 0:
                # 修复：使用实际攻击样本数计算ASR
                current_asr = 100. * bkd_correct / max(1, backdoor_samples_total)
                print(f'Train Epoch: {epoch} [{batch_idx}/{min_batches} ({100. * batch_idx / min_batches:.0f}%)]'
                      f'\tLoss: {loss.item():.6f}, Clean Acc: {current_clean_acc:.2f}%, ASR: {current_asr:.2f}%')
            else:
                print(f'Train Epoch: {epoch} [{batch_idx}/{min_batches} ({100. * batch_idx / min_batches:.0f}%)]'
                      f'\tLoss: {loss.item():.6f}, Clean Acc: {current_clean_acc:.2f}%')
        
        monitor_gpu_usage(epoch, batch_idx)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / min_batches
    accuracy = 100. * clean_correct / total
    
    # Calculate attack success rate
    attack_success_rate = 0.0
    if enable_backdoor and bkd_correct > 0:
        # 修复：使用实际攻击样本数计算ASR
        attack_success_rate = min(100.0, 100.0 * bkd_correct / max(1, backdoor_samples_total))
    
    # Calculate inference accuracy
    inference_accuracy = 0
    if has_inference and label_inference_module and label_inference_module.initialized:
        # Simplified inference accuracy calculation
        if hasattr(label_inference_module, 'accuracy_history') and len(label_inference_module.accuracy_history) > 0:
            inference_accuracy = label_inference_module.accuracy_history[-1]
    
    return avg_loss, accuracy, attack_success_rate, inference_accuracy

def test(modelC, bottom_models, party_test_loaders, is_backdoor=False, epoch=0, args=None):
    """Test model performance"""
    modelC.eval()
    for model in bottom_models:
        model.eval()
    
    test_loss = 0
    clean_correct = 0
    bkd_correct = 0
    backdoor_samples = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    # Get minimum data loader length
    min_batches = min(len(loader) for loader in party_test_loaders)
    
    # Check if backdoor attack should be active
    should_test_backdoor = is_backdoor and epoch >= args.Ebkd
    
    with torch.no_grad():
        for batch_idx in range(min_batches):
            # Collect data from all parties
            party_data = []
            party_labels = []
            
            for party_id, loader in enumerate(party_test_loaders):
                data_iter = iter(loader)
                for _ in range(batch_idx + 1):
                    try:
                        data, target = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        data, target = next(data_iter)
                
                data, target = data.to(DEVICE), target.to(DEVICE)
                party_data.append(data)
                if party_id == 0:
                    party_labels = target
            
            total += len(party_labels)
            
            # Forward propagation - clean data
            bottom_outputs_clean = []
            for model, data in zip(bottom_models, party_data):
                output = model(data)
                bottom_outputs_clean.append(output)
            
            combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
            output_clean = modelC(combined_output_clean)
            
            # Calculate loss
            test_loss += criterion(output_clean, party_labels).item()
            
            # Prediction
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            clean_correct += pred_clean.eq(party_labels.view_as(pred_clean)).sum().item()
            
            # Only test backdoor attack if epoch >= Ebkd
            if should_test_backdoor:
                # Prepare backdoor data
                attack_flags = torch.zeros(len(party_labels), dtype=torch.bool).to(DEVICE)
                attack_portion = int(len(party_labels) * args.poison_budget)
                if attack_portion > 0:
                    attack_flags[:attack_portion] = True
                    backdoor_samples += attack_flags.sum().item()
                    
                    bkd_target = party_labels.clone()
                    bkd_target[attack_flags] = args.target_class
                    
                    # Forward propagation - backdoor data
                    bottom_outputs_bkd = []
                    for i, (model, data) in enumerate(zip(bottom_models, party_data)):
                        if i == args.bkd_adversary:
                            output = model(data, attack_flags=attack_flags)
                        else:
                            output = model(data)
                        bottom_outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                    output_bkd = modelC(combined_output_bkd)
                    
                    pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                    bkd_correct += pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
    
    # Calculate results
    test_loss /= total
    accuracy = 100. * clean_correct / total
    
    attack_success_rate = 0.0
    if backdoor_samples > 0:
        attack_success_rate = min(100.0, 100.0 * bkd_correct / backdoor_samples)
    
    # Print results
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}% ({clean_correct}/{total})')
    if should_test_backdoor:
        print(f'Backdoor ASR: {attack_success_rate:.2f}% ({bkd_correct}/{backdoor_samples})')
    elif is_backdoor:
        print(f'Backdoor attack not active yet (epoch {epoch} < {args.Ebkd})')
    
    return test_loss, accuracy, attack_success_rate, 0

def main():
    import sys
    print("[START] VILLAIN training...", flush=True)
    sys.stdout.flush()
    
    # Set random seed
    print("[INIT] Setting random seeds...", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print("[INIT] Random seeds set", flush=True)
    
    # Set device
    print("[INIT] Setting up device...", flush=True)
    DEVICE = set_device()
    print(f"[INIT] Device set to: {DEVICE}", flush=True)
    
    # Verify GPU usage
    print("[INIT] Verifying GPU setup...", flush=True)
    print("\n" + "="*50, flush=True)
    print("Verifying GPU usage...", flush=True)
    gpu_available = verify_gpu_usage()
    print("="*50, flush=True)
    print("[INIT] GPU verification completed", flush=True)
    
    # Create checkpoint directory
    print("[INIT] Creating checkpoint directory...", flush=True)
    try:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"[INIT] Checkpoint directory created: {args.checkpoint_dir}", flush=True)
    except Exception as e:
        print(f"[INIT] Failed to create checkpoint directory: {e}", flush=True)
    
    print("[INIT] Starting training setup...", flush=True)
    print("\n" + "="*80, flush=True)
    print(f"VILLAIN Attack Training (with Label Inference) - Dataset: {args.dataset}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Number of parties: {args.party_num}", flush=True)
    print(f"Malicious party ID: {args.bkd_adversary}", flush=True)
    print(f"Target class: {args.target_class}", flush=True)
    print(f"Backdoor start epoch: {args.Ebkd}", flush=True)
    print("="*80 + "\n", flush=True)

    # Load dataset
    print("[LOAD] Loading Bank Marketing dataset...")
    train_loader, test_loader, feature_dim = load_bank_dataset(args.data_dir, args.batch_size)
    print("[LOAD] Dataset loaded successfully!")
    
    # Create party data loaders
    print("[LOAD] Creating party data loaders...")
    party_train_loaders, party_test_loaders = create_party_data_loaders(train_loader, test_loader)
    print("[LOAD] Party data loaders created!")

    # Create models
    print("[MODEL] Creating models...", flush=True)
    bottom_models, modelC = create_bank_models(feature_dim)
    print("[MODEL] Models created successfully!", flush=True)

    # Create optimizers
    print("[OPT] Creating optimizers...", flush=True)
    if args.use_adam:
        print("[OPT] Using Adam optimizer", flush=True)
        optimizers = [optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
                     for model in bottom_models]
        optimizerC = optim.Adam(modelC.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Using Adam optimizer", flush=True)
    else:
        print("[OPT] Using SGD optimizer", flush=True)
        optimizers = [optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
                     for model in bottom_models]
        optimizerC = optim.SGD(modelC.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print("Using SGD optimizer", flush=True)
    print("[OPT] Optimizers created successfully", flush=True)

    # Learning rate schedulers
    print("[OPT] Creating learning rate schedulers...", flush=True)
    if args.lr_schedule == 'cosine':
        print("[OPT] Creating cosine annealing schedulers...", flush=True)
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01) 
                     for optimizer in optimizers]
        schedulerC = optim.lr_scheduler.CosineAnnealingLR(optimizerC, T_max=args.epochs, eta_min=args.lr*0.01)
    elif args.lr_schedule == 'step':
        print("[OPT] Creating step schedulers...", flush=True)
        schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) 
                     for optimizer in optimizers]
        schedulerC = optim.lr_scheduler.StepLR(optimizerC, step_size=20, gamma=0.5)
    else:
        print("[OPT] Creating plateau schedulers...", flush=True)
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5) 
                     for optimizer in optimizers]
        schedulerC = optim.lr_scheduler.ReduceLROnPlateau(optimizerC, mode='max', factor=0.5, patience=5)
    print("[OPT] Learning rate schedulers created successfully", flush=True)

    print(f"\n[TRAIN] Starting Bank Marketing VILLAIN Attack Training", flush=True)
    print(f"[TRAIN] Epochs: {args.epochs}, Batch size: {args.batch_size}", flush=True)

    # Training loop
    print("[TRAIN] Initializing training variables...", flush=True)
    best_accuracy = 0
    best_asr = 0
    best_inference_acc = 0  # 添加最佳推理准确率跟踪
    best_epoch = 0
    no_improvement_count = 0
    print("[TRAIN] Training variables initialized", flush=True)
    
    print("[TRAIN] Starting epoch loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\n[EPOCH] {'='*20} Epoch {epoch}/{args.epochs} {'='*20}", flush=True)
        
        # Training
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            modelC, bottom_models, party_train_loaders, optimizers, optimizerC, epoch, args
        )

        # Testing
        test_loss, test_acc, _, test_inference_acc = test(
            modelC, bottom_models, party_test_loaders, is_backdoor=False, epoch=epoch, args=args
        )

        # Backdoor testing
        bkd_loss, bkd_acc, true_asr, bkd_inference_acc = test(
            modelC, bottom_models, party_test_loaders, is_backdoor=True, epoch=epoch, args=args
        )
        
        # Update learning rate
        if args.lr_schedule in ['cosine', 'step']:
            for scheduler in schedulers:
                scheduler.step()
            schedulerC.step()
        else:
            for scheduler in schedulers:
                scheduler.step(test_acc)
            schedulerC.step(test_acc)

        print(f"\nEpoch {epoch} Results:")
        print(f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%, ASR {train_asr:.2f}%, Inference Acc {train_inference_acc:.2f}%")
        print(f"Test: Loss {test_loss:.4f}, Acc {test_acc:.2f}%")
        if epoch >= args.Ebkd:
            print(f"Backdoor: ASR {true_asr:.2f}%")
        else:
            print(f"Backdoor attack not active yet (starts at epoch {args.Ebkd})")

        # Check if this is the best model
        # Only consider ASR in combined score if backdoor attack has started
        if epoch >= args.Ebkd:
            combined_score = 0.6 * test_acc + 0.4 * true_asr
            best_combined_score = 0.6 * best_accuracy + 0.4 * best_asr
        else:
            # Before backdoor attack starts, only consider clean accuracy
            combined_score = test_acc
            best_combined_score = best_accuracy
        
        if combined_score > best_combined_score:
            best_accuracy = test_acc
            best_asr = true_asr
            best_inference_acc = train_inference_acc
            best_epoch = epoch
            no_improvement_count = 0
            print(f"\nSaving best model (Epoch {epoch})")
        else:
            no_improvement_count += 1
            
        # Early stopping check - but not before backdoor attack starts
        if args.early_stopping and epoch >= args.Ebkd and no_improvement_count >= args.patience:
            print(f"\nEarly stopping triggered! Best model at Epoch {best_epoch}")
            break
    
    print("\n" + "="*60)
    print(f"Training completed! Best model (Epoch {best_epoch}):")
    print(f"Clean Accuracy: {best_accuracy:.2f}%")
    print(f"Attack Success Rate: {best_asr:.2f}%")
    print(f"Inference Accuracy: {best_inference_acc:.2f}%")
    print("="*60)

if __name__ == '__main__':
    main() 