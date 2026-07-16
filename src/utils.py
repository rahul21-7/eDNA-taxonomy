import numpy as np

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
