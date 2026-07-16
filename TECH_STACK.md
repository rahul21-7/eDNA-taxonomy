# Deep-Sea eDNA Platform: Tech Stack & Architecture Guide

A concise guide to the technical architecture, algorithms, and key concepts behind the **Deep-Sea eDNA Analysis Platform** for Smart India Hackathon.

---

## 1. System Architecture & Tech Stack

* **Frontend:** Serves a single-page HTML5/Vanilla CSS & JS dashboard. Uses **Chart.js** for biodiversity visuals (doughnut and bar charts).
* **Backend:** Built with **Flask** and **Flask-CORS** (serves routes and handles REST requests).
* **Bioinformatics:** Uses **Biopython (`Bio.SeqIO`)** to parse FASTA/FASTQ sequence inputs directly from in-memory string streams.
* **Deep Learning Engine:** Uses **TensorFlow / Keras** to load model weights and execute predictions.
* **Utilities:** Uses **NumPy** for numerical formatting, **joblib** to load label encoders/metadata, and **scikit-learn** for decoding classes.

---

## 2. Sequence Preprocessing Pipeline

Raw genomic sequence inputs are preprocessed in four key steps:
1. **Parsing:** Biopython's `SeqIO` extracts raw nucleotide sequences. Analysis is capped at 1000 sequences for performance.
2. **Ambiguous Base Handling:** A regex replaces any non-standard nucleotides (non-ACGT) with `N` (unknown base).
3. **Length Standardization:** Sequences are standardized to **exactly 1500 bp**:
   * *Longer sequences* are centrally cropped to preserve the hypervariable center.
   * *Shorter sequences* are padded at the end with `N` characters.
4. **Numerical Encoding:** Characters are converted to integers using a static vocabulary map:
   $$\text{'N'} \rightarrow 0, \quad \text{'A'} \rightarrow 1, \quad \text{'C'} \rightarrow 2, \quad \text{'G'} \rightarrow 3, \quad \text{'T'} \rightarrow 4$$
   Yields numerical input tensors of shape `(batch_size, 1500)`.

---

## 3. Model Architecture: Hybrid CNN-BiLSTM

Instead of a heavy BERT-style Transformer, the platform utilizes a **hybrid Multi-Scale CNN-BiLSTM architecture** (275k parameters, ~1MB) for sequence classification.

```text
Input (1500,) ──► Embedding (64) ──┬──► Conv1D (k=3) ──► MaxPooling ──┐
                                    ├──► Conv1D (k=5) ──► MaxPooling ──┼──► Concatenate (750, 256) ──► BiLSTM (128) ──► Dense Head
                                    └──► Conv1D (k=7) ──► MaxPooling ──┘
```

### Key Components & Roles:
1. **CNN for Feature Extraction:** Three parallel 1D Convolutional branches (`Conv1D` with kernel sizes **3, 5, and 7**) act as local feature extractors. They scan the sequence in parallel to capture 3-mer, 5-mer, and 7-mer local motif distributions.
2. **BiLSTM for Feature Mapping:** A Bidirectional LSTM layer takes the concatenated local features and maps them to a sequential temporal context. It processes inputs forward and backward to model relationships between hypervariable loops.
3. **Computational Efficiency:** 
   * CNNs use shared weights, minimizing parameters.
   * The CNN's MaxPooling downsamples sequence length by 50% (from 1500 to 750) before reaching the BiLSTM, drastically reducing recurrent computation.
   * The entire model footprint is **~1 MB** (compared to >300 MB for a BERT model), enabling real-time CPU inference in milliseconds.

---

## 4. Key Mathematical Algorithms

### 1. Biodiversity Shannon-Wiener Index ($H'$)
Measures species richness (number of species) and evenness (how evenly distributed they are):
$$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$
* $S$ = Total unique identified taxonomic classes.
* $p_i$ = Proportion of high-confidence reads belonging to taxon $i$ ($p_i = n_i / N_{\text{total}}$).
* *Stability:* Uses a smoothing term ($10^{-10}$) to avoid logarithmic division-by-zero errors.

### 2. Novelty Index
Used to detect new organisms not present in the SILVA 18S reference database:
* **Thresholding:** Predictions with classification confidence $< 70\%$ are flagged as **Novel Sequence Candidates**.
* **Novelty Score:** Calculated as:
  $$\text{Novelty Score} = 1 - \text{Average Confidence}$$
  Higher score indicates a high rate of uncharacterized organisms in the sample.
