# 404 Specie Not Found: Deep-Sea eDNA Platform

An AI-driven analysis platform designed to process environmental DNA (eDNA) sequencing data for biodiversity assessment in deep-sea ecosystems. This application leverages a pre-trained deep learning classifier to identify eukaryotic species from 18S rRNA marker gene sequences, calculates community biodiversity metrics, and identifies potentially novel taxonomic groups.

Developed as a solution for the **Smart India Hackathon (SIH)**, Problem Statement **SIH25042** (issued by the **Centre for Marine Living Resources and Ecology (CMLRE)**, Ministry of Earth Sciences).

---

## Key Features

* **Biological File Parsing:** Native support for FASTA, FASTQ, and compressed `.gz` formats via Biopython.
* **Deep Learning Taxonomy Classifier:** Utilizes a pre-trained Keras model trained on 18S rRNA SILVA datasets to predict taxonomic categories.
* **Biodiversity Analytics:** Computes the **Shannon-Wiener Diversity Index**, determines dominant taxa, and details phylum composition.
* **Novel Taxa Detection:** Flags sequence predictions below a confidence threshold to help researchers locate novel and uncharacterized deep-sea organisms.
* **Modern Web Interface:** A highly visual, responsive dashboard with a deep-sea theme featuring interactive charts (powered by Chart.js) and intuitive drag-and-drop file upload.
* **REST API:** Fully functional endpoints to interface with the AI pipeline programmatically.

---

## Project Structure

```
SIH-eDNA-taxonomy/
├── app.py                             # Flask API server & ML preprocessing/inference logic
├── requirements.txt                   # Project package dependencies
├── data/
│   └── 16S_sequences.fasta            # Sample sequence file (for testing)
├── models/
│   ├── 18s_trained_eDNA_model_silva.h5 # Trained TensorFlow Keras model
│   ├── 18s_label_encoder.pkl          # Scikit-learn Label Encoder for taxonomic classes
│   └── 18s_class_info.pkl             # Serialization metadata for the trained model classes
├── static/
│   └── index.html                     # Premium web interface (dashboard, charts, upload UI)
└── uploads/                           # Directory to store uploaded files
```

---

## Quick Start Guide

Follow these steps to set up a virtual environment and run the application locally on your machine.

### Prerequisites
* **Python 3.8 to 3.11** (recommended version range for TensorFlow 2.13.0 stability)
* Web browser (Chrome, Firefox, Edge, etc.)

### 1. Set Up the Workspace
Navigate to the directory containing the project:
```bash
cd "SIH-eDNA-taxonomy"
```

### 2. Create a Virtual Environment (venv)
Creating a virtual environment ensures that the required package versions do not conflict with your global Python installation.

* **Windows**:
  ```bash
  python -m venv venv
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  ```

### 3. Activate the Virtual Environment
Activate the environment to start using the localized Python workspace:

* **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
* **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```
* **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

Once activated, your terminal prompt should display `(venv)`.

### 4. Install Dependencies
Install all required libraries within the activated virtual environment:
```bash
pip install -r requirements.txt
```

### 5. Launch the Backend Server
Start the Flask application:
```bash
python app.py
```
Upon startup, the script will load the pre-trained deep-sea 18S rRNA classification model and serialize helper artifacts. By default, it runs on `http://localhost:5000/`.

### 6. Open the Web Dashboard
Since the Flask server serves the static frontend directly from the root path:
* Simply open your web browser and navigate to: **`http://localhost:5000/`**
* Alternatively, you can open `static/index.html` directly in your browser (our dynamic API client will automatically fall back to hitting the local Flask server at port 5000).

### 7. Run an Analysis
1. Select or drag-and-drop a FASTA/FASTQ file (you can use the provided [16S_sequences.fasta](data/16S_sequences.fasta) or any 18S eukaryotic dataset).
2. Configure **Analysis Parameters** (e.g. adjustment of the **Confidence Threshold** slider).
3. Click **Analyze eDNA Data**.
4. The dashboard will animate the processing stages and render the final results, community diagrams, and biodiversity indexes.

To deactivate the virtual environment when you are done, simply run:
```bash
deactivate
```


---

## API Documentation

The backend server exposes the following RESTful API endpoints:

### `POST /api/analyze`
Submits a sequence file for taxonomic prediction and biodiversity analysis.
* **Form-data Parameters:**
  * `file`: The sequencing file (FASTA/FASTQ)
  * `confidence_threshold`: Minimum probability required to accept a prediction (default: `0.7`)
  * `marker_gene`: Target gene used (e.g. `18S`, `COI`)
* **Response:** JSON object containing sequence statistics, taxonomic counts, Shannon index, list of dominant species, and data structured for Chart.js visualizations.

### `GET /api/health`
Performs a check on backend availability and confirms if the TensorFlow model is initialized.

### `GET /api/model_info`
Exposes metadata of the model, including the total number of classes trained and a list of target taxa.

---

## Reference Databases and Data Sources

The machine learning models and reference taxonomies utilized in this platform are based on the following publicly available biological data repositories:

* **SILVA Ribosomal RNA Database**: Used to train the pre-trained classification model (`18s_trained_eDNA_model_silva.h5`). SILVA provides high-quality alignment and taxonomic datasets for small subunit (16S/18S, SSU) and large subunit (23S/28S, LSU) ribosomal RNA.
  * Website: [https://www.arb-silva.de](https://www.arb-silva.de)
* **PR2 (Protist Ribosomal Reference) Database**: Provides curated reference databases for eukaryotic 18S rRNA sequences.
  * Website: [https://pr2-database.org](https://pr2-database.org)
* **NCBI SRA (Sequence Read Archive)**: Used to search, fetch, and download raw environmental DNA and metabarcoding sequencing datasets.
  * Website: [https://www.ncbi.nlm.nih.gov/sra](https://www.ncbi.nlm.nih.gov/sra)

---

## Tech Stack Explanation

For an in-depth breakdown of the architecture, machine learning model details, preprocessing workflow, and equations used for diversity calculations, please refer to the detailed guide:

**[Tech Stack and Architecture Explanation](TECH_STACK.md)**

