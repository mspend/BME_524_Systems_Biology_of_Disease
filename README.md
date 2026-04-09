### A compilation of code I wrote for my Systems Biology of Disease class (BME 524), taken Spring 2026 at ASU from Dr. Chris Plaisier. 

# Course learning outcomes
## Students will be able to:
- Design omics studies that provide ample statistical power for downstream analyses
- Identify the correct quantitative systems biology approach to analyze omics datasets: clustering, classification, enrichment analysis, network construction, or trajectory analysis
- Use quantitative reasoning to interpret systems biology analyses
- Understand how to turn interpretations into experimental studies that validate hypotheses

# Module Topics and Skills
- Modules 1-10 were lectures with no coding component, covering these topics: systems biology, complex systems, study design, statistical power, sequencing, genomics, epigenomics, transcriptomics, proteomics, and metabolomics. For each of the omics lectures, we discussed the types of experiments performed, the information they provide, and the advantages and disadvantages of each.
- Modules 11-12: Defining cell types in scRNA-seq (clustering)
    1. Explain how scRNA-seq data are generated and processed, from FASTQ files to a gene × cell count matrix.
    2. Perform quality control and preprocessing in Scanpy, including filtering low-quality cells, normalizing counts, and regressing out confounders.
    3. Conduct dimensionality reduction and clustering (PCA, UMAP, Leiden) to identify cellular subpopulations.
    4. Identify and interpret marker genes using differential expression analysis to assign biological identities to clusters.
    5. Apply reproducible Python workflows to analyze and visualize PBMC heterogeneity in a systems biology context.
- Modules 13-16: Biomarker discovery (classification)
    1. Define a biomarker and distinguish among major biomarker categories (risk, diagnostic, prognostic, therapeutic response).
    2. Describe characteristics of an ideal biomarker, including sensitivity, specificity, reproducibility, and clinical feasibility.
    3. Explain how omics and systems-level data support biomarker discovery, particularly in complex diseases.
    4. Apply supervised machine learning concepts to biomarker classification problems, including training/testing splits and model evaluation.
    5. Interpret classifier performance metrics (e.g., ROC AUC, precision, recall, PPV/NPV) in a clinical decision-making context.
- Modules 17-19: Making sense of overlaps (hypergeometric enrichment analysis)
    1. Apply hypergeometric enrichment analysis to compare different sets of the same type of omics data.
    2. Identify statistically significant overlap and correct for multiple hypothesis testing.
    3. Use hypergeometric enrichment analysis to validate established relationships and forge a link between new concepts.