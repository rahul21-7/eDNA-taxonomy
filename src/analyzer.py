import os
import re
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from Bio import SeqIO

class eDNAAnalyzer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.class_info = None
        self.max_sequence_length = 1500
        self.vocab = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}

    def load_trained_model(self):
        """Load your pre-trained model and artifacts"""
        try:
            # Load the trained model
            print("Loading pre-trained 18S rRNA model...")
            self.model = keras.models.load_model(os.path.join('models', '18s_trained_eDNA_model_silva.h5'))
            print("[OK] Model loaded successfully!")

            # Load label encoder
            self.label_encoder = joblib.load(os.path.join('models', '18s_label_encoder.pkl'))
            print("[OK] Label encoder loaded successfully!")

            # Load class information
            self.class_info = joblib.load(os.path.join('models', '18s_class_info.pkl'))
            print("[OK] Class information loaded successfully!")

            print(f"[OK] Model trained on {len(self.label_encoder.classes_)} taxonomic classes")
            print(f"[OK] Classes: {list(self.label_encoder.classes_)[:10]}...")  # Show first 10

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def preprocess_sequences(self, sequences):
        """Preprocess sequences using the same method as training"""
        processed_seqs = []

        for seq in sequences:
            # Clean sequence
            clean_seq = self.remove_ambiguous_bases(seq)

            # Pad sequence
            padded_seq = self.pad_sequence(clean_seq)

            # Convert to numerical
            numerical_seq = [self.vocab.get(base, 0) for base in padded_seq]
            processed_seqs.append(numerical_seq)

        return np.array(processed_seqs)

    def remove_ambiguous_bases(self, sequence):
        """Remove or replace ambiguous bases"""
        sequence = re.sub(r'[^ACGT]', 'N', sequence.upper())
        return sequence

    def pad_sequence(self, sequence):
        """Pad sequence to uniform length"""
        if len(sequence) > self.max_sequence_length:
            # Take middle section for 18S
            start = (len(sequence) - self.max_sequence_length) // 2
            padded_seq = sequence[start:start + self.max_sequence_length]
        else:
            # Pad if too short
            padded_seq = sequence + 'N' * (self.max_sequence_length - len(sequence))
        return padded_seq

    def predict_taxonomy(self, sequences, confidence_threshold=0.7):
        """Predict taxonomic classification for sequences"""
        if self.model is None:
            self.load_trained_model()

        # Preprocess sequences
        X = self.preprocess_sequences(sequences)

        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)

        # Convert to class names
        taxonomic_labels = self.label_encoder.inverse_transform(predicted_classes)

        # Get top predictions for each sequence
        top_predictions = []
        for i in range(len(predictions)):
            top3_indices = np.argsort(predictions[i])[-3:][::-1]
            top3 = {
                self.label_encoder.classes_[idx]: float(predictions[i][idx])
                for idx in top3_indices
            }
            top_predictions.append(top3)

        return taxonomic_labels, confidence_scores, top_predictions


def parse_sequences(file_content, filename):
    """Parse FASTA/FASTQ files and extract sequences"""
    sequences = []
    headers = []

    try:
        # Determine file type
        file_extension = filename.split('.')[-1].lower()

        if file_extension in ['fasta', 'fa', 'fna']:
            format_type = 'fasta'
        elif file_extension in ['fastq', 'fq']:
            format_type = 'fastq'
        elif file_extension == 'gz':
            # For gzipped files, assume FASTA
            format_type = 'fasta'
        else:
            return sequences, headers

        # Parse sequences
        file_like = io.StringIO(file_content.decode('utf-8'))

        for record in SeqIO.parse(file_like, format_type):
            sequences.append(str(record.seq))
            headers.append(record.description)

    except Exception as e:
        print(f"Error parsing sequences: {e}")

    return sequences, headers
