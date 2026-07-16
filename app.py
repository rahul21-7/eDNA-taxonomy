from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Bio import SeqIO
import io
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter
import tempfile

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)


@app.route('/')
def index():
    """Serve the frontend index.html page"""
    return app.send_static_file('index.html')


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


# Initialize the analyzer
analyzer = eDNAAnalyzer()


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


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def calculate_biodiversity_metrics(predictions, confidence_scores, confidence_threshold=0.7):
    """Calculate biodiversity metrics from predictions"""
    # Filter by confidence
    high_conf_mask = confidence_scores >= confidence_threshold
    filtered_predictions = predictions[high_conf_mask]

    if len(filtered_predictions) == 0:
        return {
            'taxa_counts': {},
            'shannon_index': 0,
            'dominant_taxa': [],
            'high_confidence_count': 0,
            'total_count': len(predictions),
            'total_taxa_identified': 0
        }

    # Count taxa
    unique_taxa, counts = np.unique(filtered_predictions, return_counts=True)

    # Calculate Shannon diversity index
    proportions = counts / counts.sum()
    shannon_index = -np.sum(proportions * np.log(proportions + 1e-10))  # Add small value to avoid log(0)

    # Get dominant taxa (top 10)
    taxa_with_counts = list(zip(unique_taxa, counts))
    taxa_with_counts.sort(key=lambda x: x[1], reverse=True)
    dominant_taxa = taxa_with_counts[:10]

    # Count by taxonomic level (simplified)
    phylum_counts = {}
    for taxon in filtered_predictions:
        # Simple heuristic: if taxon contains specific words, categorize
        if 'Other_Rare_Eukaryotes' in taxon:
            phylum = 'Rare Eukaryotes'
        elif any(word in taxon for word in ['Protist', 'Algae', 'Diatom']):
            phylum = 'Protists'
        elif any(word in taxon for word in ['Fungi', 'Yeast']):
            phylum = 'Fungi'
        elif any(word in taxon for word in ['Animal', 'Metazoa']):
            phylum = 'Animals'
        elif any(word in taxon for word in ['Plant', 'Archaeplastida']):
            phylum = 'Plants'
        else:
            phylum = 'Other Eukaryotes'

        phylum_counts[phylum] = phylum_counts.get(phylum, 0) + 1

    # Convert to serializable types
    return {
        'taxa_counts': dict(zip(unique_taxa.tolist(), counts.tolist())),
        'phylum_composition': phylum_counts,
        'shannon_index': float(shannon_index),
        'dominant_taxa': [(taxon, int(count)) for taxon, count in dominant_taxa],
        'high_confidence_count': int(len(filtered_predictions)),
        'total_count': int(len(predictions)),
        'total_taxa_identified': int(len(unique_taxa))
    }


def generate_visualization_data(taxonomic_labels, confidence_scores, confidence_threshold=0.7):
    """Generate data for frontend visualizations"""
    # Filter by confidence
    high_conf_mask = confidence_scores >= confidence_threshold
    filtered_labels = taxonomic_labels[high_conf_mask]

    if len(filtered_labels) == 0:
        return {
            'phylum_composition': {'No high-confidence predictions': 1},
            'level_diversity': {'Phylum': 0, 'Class': 0, 'Order': 0, 'Family': 0, 'Genus': 0, 'Species': 0}
        }

    # Count by phylum (simplified)
    phylum_composition = {}
    for taxon in filtered_labels:
        if 'Other_Rare_Eukaryotes' in taxon:
            phylum = 'Rare Eukaryotes'
        elif any(word in taxon for word in ['Protist', 'Algae', 'Diatom']):
            phylum = 'Protists'
        elif any(word in taxon for word in ['Fungi', 'Yeast']):
            phylum = 'Fungi'
        elif any(word in taxon for word in ['Animal', 'Metazoa']):
            phylum = 'Animals'
        elif any(word in taxon for word in ['Plant', 'Archaeplastida']):
            phylum = 'Plants'
        else:
            phylum = 'Other Eukaryotes'

        phylum_composition[phylum] = phylum_composition.get(phylum, 0) + 1

    # Simulate diversity at different taxonomic levels
    unique_taxa = len(np.unique(filtered_labels))
    level_diversity = {
        'Phylum': int(min(unique_taxa, 10)),  # Simplified estimation
        'Class': int(min(unique_taxa * 2, 25)),
        'Order': int(min(unique_taxa * 3, 50)),
        'Family': int(min(unique_taxa * 4, 75)),
        'Genus': int(min(unique_taxa * 5, 100)),
        'Species': int(min(unique_taxa * 6, 150))
    }

    return {
        'phylum_composition': phylum_composition,
        'level_diversity': level_diversity
    }


@app.route('/api/analyze', methods=['POST'])
def analyze_edna():
    """Main endpoint for eDNA analysis using your trained model"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get analysis parameters
        analysis_type = request.form.get('analysis_type', 'taxonomy')
        marker_gene = request.form.get('marker_gene', '18S')
        database = request.form.get('database', 'custom')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.7))

        # Read file content
        file_content = file.read()

        # Parse sequences
        sequences, headers = parse_sequences(file_content, file.filename)

        if not sequences:
            return jsonify({'error': 'No valid sequences found in file'}), 400

        print(f"Processing {len(sequences)} sequences...")

        # For large files, use a subset to speed up processing
        if len(sequences) > 1000:
            sequences = sequences[:1000]
            print(f"Using first 1000 sequences for analysis")

        # Perform taxonomic classification using YOUR trained model
        taxonomic_labels, confidence_scores, top_predictions = analyzer.predict_taxonomy(
            sequences, confidence_threshold
        )

        # Calculate biodiversity metrics
        biodiversity = calculate_biodiversity_metrics(
            taxonomic_labels, confidence_scores, confidence_threshold
        )

        # Generate visualization data
        visualization_data = generate_visualization_data(
            taxonomic_labels, confidence_scores, confidence_threshold
        )

        # Calculate sequence statistics
        seq_lengths = [len(seq) for seq in sequences]
        avg_length = np.mean(seq_lengths)
        avg_quality = np.mean(confidence_scores)

        # Count novel taxa (low confidence predictions)
        low_confidence_count = np.sum(confidence_scores < confidence_threshold)

        # Prepare response - Convert all numpy types to native Python types
        results = {
            'summary': {
                'total_sequences': int(len(sequences)),
                'high_confidence_sequences': int(biodiversity['high_confidence_count']),
                'average_length': float(round(avg_length, 1)),
                'average_quality': float(round(avg_quality, 3)),
                'sequences_analyzed': int(len(sequences))
            },
            'biodiversity': {
                'taxa_counts': biodiversity['taxa_counts'],
                'shannon_index': biodiversity['shannon_index'],
                'dominant_taxa': biodiversity['dominant_taxa'],
                'total_taxa_identified': biodiversity['total_taxa_identified'],
                'high_confidence_count': biodiversity['high_confidence_count']
            },
            'novel_taxa': {
                'novel_sequence_count': int(low_confidence_count),
                'novelty_score': float(round(1 - avg_quality, 3)),
                'low_confidence_sequences': int(low_confidence_count)
            },
            'visualization_data': visualization_data,
            'analysis_parameters': {
                'analysis_type': analysis_type,
                'marker_gene': marker_gene,
                'database': database,
                'confidence_threshold': float(confidence_threshold),
                'model_used': '18S_rRNA_Classifier_v1.0'
            },
            'sample_predictions': [
                {
                    'header': headers[i] if i < len(headers) else f"Sequence_{i + 1}",
                    'predicted_taxon': str(taxonomic_labels[i]),
                    'confidence': float(confidence_scores[i]),
                    'sequence_length': int(len(sequences[i]))
                }
                for i in range(min(5, len(sequences)))  # Include first 5 predictions as sample
            ]
        }

        # Convert all numpy types to native Python types
        results = convert_to_serializable(results)

        print(f"[OK] Analysis complete: {biodiversity['total_taxa_identified']} taxa identified")
        return jsonify(results)

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if analyzer.model is None:
            analyzer.load_trained_model()

        return jsonify({
            'status': 'healthy',
            'model_loaded': analyzer.model is not None,
            'model_classes': int(len(analyzer.label_encoder.classes_)) if analyzer.label_encoder else 0,
            'model_name': '18S rRNA Deep-Sea eDNA Classifier'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    try:
        if analyzer.model is None:
            analyzer.load_trained_model()

        model_info = {
            'model_name': '18S rRNA eDNA Classifier',
            'classes_count': int(len(analyzer.label_encoder.classes_)),
            'classes': list(analyzer.label_encoder.classes_),
            'sequence_length': int(analyzer.max_sequence_length),
            'training_info': analyzer.class_info if analyzer.class_info else {}
        }

        # Convert to serializable
        model_info = convert_to_serializable(model_info)

        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify the API is working"""
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly',
        'model_loaded': analyzer.model is not None
    })


if __name__ == '__main__':
    # Load the trained model on startup
    print("Initializing 404 Specie Not Found Analysis Platform...")
    print("Loading pre-trained 18S rRNA model...")

    try:
        analyzer.load_trained_model()
        print("[OK] Model loaded successfully!")
        print("[OK] 404 Specie Not Found ready!")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print("[ERROR] Platform started in fallback mode")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)