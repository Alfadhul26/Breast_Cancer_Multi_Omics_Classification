# Breast Cancer Molecular Subtype Classification Using Multi-Omics Data

**A machine-learning project integrating gene expression, copy-number variation, and somatic mutation profiles to classify four PAM50 breast cancer molecular subtypes, with independent evaluation on TCGA-BRCA and an interactive Streamlit application.**

## Overview and motivation

Breast cancer is molecularly heterogeneous: tumours with similar clinical presentations can differ in their underlying biology and outcomes. Molecular subtyping helps characterise this heterogeneity, but high-dimensional molecular measurements introduce challenges for statistical learning, feature selection, and generalisation across cohorts.

This project investigates whether combining three molecular data layers improves prediction of **Basal, HER2-enriched, Luminal A, and Luminal B** subtypes. Particular attention is given to feature selection, comparison of modelling approaches, and performance on a cohort not used for model development.

**Research question:** How effectively can supervised machine-learning models classify PAM50 breast cancer subtypes from individual and integrated multi-omics profiles, and how well do these models generalise to an external cohort?

## Data

The analysis uses the **MOANNA** multi-omics dataset, with METABRIC samples for model development and TCGA-BRCA samples for external evaluation.

| Cohort | Role | Samples |
|---|---|---:|
| METABRIC training | Model fitting | 1,182 |
| METABRIC validation | Model comparison and selection | 507 |
| METABRIC combined | Final model refitting | 1,689 |
| TCGA-BRCA | External evaluation | 631 |

Each sample contains **46,776 molecular features**, comprising **15,592 genes per modality**:

| Modality | Suffix | Description |
|---|---|---|
| Gene expression | `_EXPR` | Normalised and standardised gene-expression measurements |
| Copy-number variation | `_CNV` | Estimated gene-level copy-number measurements |
| Somatic mutation | `_SNP` | Binary indicators of somatic mutation status |

The target is the four-class `Pam50Subtype` label: **Basal (0), HER2-enriched (1), Luminal A (2), and Luminal B (3)**.

**Data source:** MOANNA dataset, derived from METABRIC and TCGA-BRCA. **[https://zenodo.org/records/4326602]** The notebooks currently expect local MOANNA TSV files; raw datasets are not included in this repository.

## Experimental approach

The work follows three stages:

1. **Exploratory data analysis:** Verify sample/label alignment, inspect modality distributions and class balance, examine molecular features and subtype patterns, and explore dimensionality reduction.
2. **Model development:** Establish single-modality and multi-omics baselines; compare feature-selection strategies and classifiers using the METABRIC training and validation cohorts.
3. **Final model and external evaluation:** Compare soft voting with stacking, refit the selected approaches on combined METABRIC data, and evaluate once on TCGA-BRCA.

### Experiments and feature selection

The experiments include:

- **Modality comparisons:** Expression alone, CNV alone, mutation features alone, and multi-omics combinations.
- **Feature selection:** ANOVA-based `SelectKBest`, mutual-information selection, recursive feature elimination (RFE), variance filtering, and regularisation.
- **Classifiers:** Logistic regression (including L1, L2, and elastic-net variants), random forests, histogram gradient boosting (HGB), and support vector machines.
- **Additional analyses:** PCA-based representations and experiments excluding available PAM50-associated expression features.

Feature selection and scaling are incorporated into fitted scikit-learn pipelines for the final estimators, so they can be learned from the fitting data rather than applied manually to the external test cohort.

## Final model architecture

The deployed model is a **stacking ensemble** with two base estimators:

- **HistGradientBoostingClassifier:** A tree-based model for nonlinear relationships.
- **L1-regularised logistic regression:** A linear classifier with sparse coefficients.

A **logistic regression meta-classifier** combines their class-probability outputs. The stacking estimator uses **3-fold cross-validation** to generate meta-model.

The multi-omics preprocessing selects informative expression and CNV features and filters invariant SNP features.

**Training and evaluation flow**

```text
METABRIC training (n=1,182)
    │
    ├── Fit candidate pipelines and ensembles
    │
METABRIC validation (n=507)
    │
    ├── Compare candidate models
    │
Combined METABRIC (n=1,689)
    │
    ├── Refit final ensembles
    │
TCGA-BRCA (n=631)
    └── External evaluation
```

## Results

### Ensemble comparison

The results below are transcribed from the final-model notebook. All metrics are proportions (higher is better).

| Evaluation cohort | Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| METABRIC validation | Soft voting | 0.8895 | 0.8913 | 0.8905 |
| METABRIC validation | Stacking | 0.8876 | 0.8864 | 0.8836 |
| TCGA-BRCA external | Soft voting | 0.8796 | 0.8523 | 0.846 |
| TCGA-BRCA external | Stacking | 0.8764 | 0.8532 | 0.846 |

Soft voting and stacking performed similarly on the external cohort. The deployed Streamlit application uses stacking. These differences are small and should not be interpreted as evidence of superiority without uncertainty estimates or additional validation.

## Interactive application

The Streamlit application (`app.py`) provides:

- CSV upload and validation against the trained model's expected molecular features.
- Predicted PAM50 subtypes and class probabilities.
- Cohort-level subtype distribution and composition.
- Individual-sample probability exploration.
- Downloadable prediction results, retaining or generating sample identifiers as appropriate.

**Live application:** []

**Application Demo:** []

**Important:** Uploaded data must follow the same upstream normalisation and encoding conventions as the training data. The app is a **research demonstration**, not a diagnostic tool. Model probabilities are not calibrated clinical risk estimates.

## Repository structure

```text
Cancer_Subtypes_Multi_Omics_Classification/
├── app.py
├── breast_cancer_stacking.pkl
├── Notebooks/
│   └── Multi_Omics_Capstone.ipynb
├── Outputs/
│   └── Presentation.pdf
├── README.md
└── requirements.txt
```

The tree reflects the current folder layout; replace bracketed descriptions with actual filenames when publishing. A `.gitignore` is recommended to exclude local environments, caches, raw datasets, and temporary outputs.

## Limitations

- **Cross-cohort generalisation:** METABRIC and TCGA-BRCA may differ in measurement platforms, preprocessing, sample composition, and technical effects.
- **High dimensionality:** The feature space is much larger than the number of training samples, increasing overfitting risk and making robust feature selection essential.
- **Subtype label interpretation:** The model learns the provided PAM50 labels; prediction accuracy is not independent biological or clinical validation of those labels.
- **Model interpretability:** Selected genes and tree split frequencies indicate model associations, not causal drivers of disease.
- **Probability calibration:** Class probabilities have not been established as calibrated probabilities for clinical decision-making.
- **Reproducibility:** The existing research notebook uses local dataset paths; a clean rerun requires downloading the source data and adjusting those paths.

## Future work

1. Improve reproducibility by standardising data paths, environment versions, and notebook execution from a clean kernel.
2. Compare the ensemble with a multilayer perceptron (MLP) as a deep-learning baseline for high-dimensional tabular omics data.
3. Explore biologically informed feature selection and pathway-level interpretation, with appropriate external validation.

---

**Disclaimer:** This project is intended for research and educational purposes only. It must not be used for clinical diagnosis, prognosis, or treatment decisions.