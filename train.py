import os
import re
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Bio import SeqIO

# Vocabulary mapping identical to app's analyzer
VOCAB = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
MAX_SEQUENCE_LENGTH = 1500

def clean_and_pad_sequence(seq_str, max_len=MAX_SEQUENCE_LENGTH):
    """Clean ambiguous bases and crop/pad sequence to fixed length"""
    # Clean non-ACGT bases
    clean_seq = re.sub(r'[^ACGT]', 'N', seq_str.upper())
    
    # Pad or crop centrally
    if len(clean_seq) > max_len:
        start = (len(clean_seq) - max_len) // 2
        padded_seq = clean_seq[start:start + max_len]
    else:
        padded_seq = clean_seq + 'N' * (max_len - len(clean_seq))
        
    return [VOCAB.get(base, 0) for base in padded_seq]


def build_cnn_bilstm_model(max_len=MAX_SEQUENCE_LENGTH, vocab_size=5, num_classes=42):
    """Build the exact reverse-engineered CNN-BiLSTM hybrid classifier"""
    inputs = layers.Input(shape=(max_len,), name="input_layer")
    
    # Embedding layer
    x = layers.Embedding(input_dim=vocab_size, output_dim=64, name="embedding")(inputs)
    
    # Multi-scale CNN Feature Extraction branches (Motif Scanners)
    # Branch 1 (kernel size 3)
    c1 = layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv1d_3")(x)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling1D(pool_size=2)(c1)
    
    # Branch 2 (kernel size 5)
    c2 = layers.Conv1D(filters=128, kernel_size=5, padding='same', name="conv1d_5")(x)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling1D(pool_size=2)(c2)
    
    # Branch 3 (kernel size 7)
    c3 = layers.Conv1D(filters=64, kernel_size=7, padding='same', name="conv1d_7")(x)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling1D(pool_size=2)(c3)
    
    # Concat motifs
    merged = layers.Concatenate(name="concat_motifs")([p1, p2, p3])
    
    # Context Mapping (BiLSTM)
    bilstm = layers.Bidirectional(
        layers.LSTM(units=64, return_sequences=True), 
        name="bidirectional_lstm"
    )(merged)
    bilstm = layers.Dropout(0.2)(bilstm)
    
    # Classifier Head
    pool = layers.GlobalAveragePooling1D(name="global_pooling")(bilstm)
    
    dense1 = layers.Dense(128, activation="relu", name="dense_128")(pool)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.2)(dense1)
    
    dense2 = layers.Dense(64, activation="relu", name="dense_64")(dense1)
    dense2 = layers.Dropout(0.2)(dense2)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(dense2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def generate_synthetic_data(num_samples=100):
    """Generate synthetic sequence data for demo mode training"""
    sample_taxa = [
        'Actinobacteria', 'Alphaproteobacteria', 'Alveolata', 'Amoebozoa', 
        'Bacilli', 'Bacteroidia', 'Chloroplastida', 'Cyanobacteriia', 
        'Gammaproteobacteria', 'Stramenopiles', 'Other_Rare_Eukaryotes'
    ]
    
    bases = ['A', 'C', 'G', 'T', 'N']
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random length sequence
        seq_len = np.random.randint(500, 1600)
        seq = "".join(np.random.choice(bases, size=seq_len, p=[0.27, 0.23, 0.23, 0.25, 0.02]))
        sequences.append(seq)
        labels.append(np.random.choice(sample_taxa))
        
    return sequences, labels


def parse_dataset_file(file_path):
    """Parse custom training sequences from FASTA or CSV dataset"""
    sequences = []
    labels = []
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.fasta', '.fa', '.fna']:
        # For FASTA, we expect format: >SequenceID|TaxonomyLabel
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
            # Try parsing taxonomy label from header
            parts = record.description.split('|')
            if len(parts) > 1:
                labels.append(parts[-1].strip())
            else:
                # Fallback check for [class=Taxon] tag or similar
                match = re.search(r'\[class=([^\]]+)\]', record.description)
                if match:
                    labels.append(match.group(1).strip())
                else:
                    labels.append("Other_Rare_Eukaryotes")
    elif file_ext == '.csv':
        # Expect CSV with 'sequence' and 'label' columns
        import pandas as pd
        df = pd.read_csv(file_path)
        sequences = df['sequence'].tolist()
        labels = df['label'].tolist()
    else:
        raise ValueError("Unsupported training file format. Please provide .fasta or .csv file.")
        
    return sequences, labels


def main():
    parser = argparse.ArgumentParser(description="Train 18S rRNA eDNA CNN-BiLSTM Classifier")
    parser.add_argument('--input', type=str, default=None, help="Path to input training file (.fasta or .csv)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--demo', action='store_true', help="Force run demo training with synthetic data")
    
    args = parser.parse_args()
    os.makedirs('models', exist_ok=True)
    
    if args.demo or not args.input or not os.path.exists(args.input):
        print("[INFO] No input training file found or --demo specified. Running in synthetic demo mode...")
        sequences, raw_labels = generate_synthetic_data(120)
        dataset_name = "Synthetic_Demo_eDNA"
    else:
        print(f"[INFO] Ingesting training dataset from: {args.input}")
        sequences, raw_labels = parse_dataset_file(args.input)
        dataset_name = os.path.basename(args.input)
        
    print(f"[INFO] Total sequences parsed: {len(sequences)}")
    
    # 1. Preprocess sequences into numerical vectors
    print("[INFO] Vectorizing sequences...")
    X = np.array([clean_and_pad_sequence(seq) for seq in sequences])
    
    # 2. Encode labels
    print("[INFO] Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Training set size: {len(X_train)} | Validation set size: {len(X_test)}")
    print(f"[INFO] Detected {num_classes} distinct taxonomy classes.")
    
    # 4. Build and train model
    print("[INFO] Building model architecture...")
    model = build_cnn_bilstm_model(num_classes=num_classes)
    model.summary()
    
    print(f"[INFO] Starting model training for {args.epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # 5. Save model and metadata artifacts
    print("[INFO] Saving trained model and metadata artifacts...")
    model_path = os.path.join('models', '18s_trained_eDNA_model_silva.h5')
    encoder_path = os.path.join('models', '18s_label_encoder.pkl')
    info_path = os.path.join('models', '18s_class_info.pkl')
    
    # Save TensorFlow weights/model
    model.save(model_path)
    print(f"[OK] Model saved to {model_path}")
    
    # Save Scikit-Learn Label Encoder
    joblib.dump(label_encoder, encoder_path)
    print(f"[OK] Label encoder saved to {encoder_path}")
    
    # Save training metadata
    class_info = {
        'classes': [str(c) for c in label_encoder.classes_],
        'other_threshold': 10,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'dataset': dataset_name,
        'total_classes': num_classes
    }
    joblib.dump(class_info, info_path)
    print(f"[OK] Training metadata saved to {info_path}")
    print("[SUCCESS] Training pipeline complete!")


if __name__ == '__main__':
    main()
