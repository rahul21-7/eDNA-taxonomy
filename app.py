from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from analyzer import eDNAAnalyzer, parse_sequences
from utils import (
    calculate_biodiversity_metrics,
    generate_visualization_data,
    convert_to_serializable
)

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# Initialize the analyzer
analyzer = eDNAAnalyzer()


@app.route('/')
def index():
    """Serve the frontend index.html page"""
    return app.send_static_file('index.html')


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