# AI in Pharmacovigilance

A computational biology coursework project exploring statistical
signal detection, machine learning and natural language processing
for adverse drug reaction (ADR) analysis and pharmacovigilance.

## Project Overview

This repository explores computational approaches to identifying
and prioritising potential drug safety signals using the FDA
Adverse Event Reporting System (FAERS) and text-based analysis.

The work is organised into three progressive projects, moving
from statistical analysis to machine learning and NLP-based
signal prioritisation.

## 1. FAERS Signal Detection and Analysis

[View Assignment 1](./BAditya_assignment1)

- Analysis of FDA FAERS 2025 Q3 data.
- Identification of drug-adverse reaction associations.
- Statistical signal detection.
- Analysis of age and sex distributions.
- Temporal trends and frequently reported adverse reactions.
- Visualisation of pharmacovigilance signals.

Deliverables include a Jupyter notebook, report, figures
and processed results.

## 2. EBGM and Machine Learning Analysis

[View Assignment 2](./Baditya_assignment2)

- Empirical Bayes Geometric Mean (EBGM) signal analysis.
- Preparation of machine learning datasets.
- Integration of drug-related features.
- Model comparison and performance evaluation.
- Feature importance analysis.
- Ablation analysis of clinical features.
- ROC and precision-recall curve visualisation.

Deliverables include a Jupyter notebook, report, processed
datasets, model evaluation results and figures.

## 3. AI-Driven ADR Signal Detection

[View Mini Project](./Baditya_miniproject)

- Text preprocessing and narrative generation.
- Named Entity Recognition (NER).
- Drug-ADR relation extraction.
- Entity and relation extraction evaluation.
- Stress testing of extraction methods.
- Integration of text-derived signals with FAERS results.
- Drug-ADR signal prioritisation and network visualisation.
- Interactive dashboard implementation.

The text corpus used in this project is synthetic because
PubMed was inaccessible during the original implementation.
The results demonstrate the computational workflow rather
than independent validation using published literature.

## Repository Structure

    faers-pharmacovigilance/
    |-- BAditya_assignment1/
    |-- Baditya_assignment2/
    |-- Baditya_miniproject/
    |-- .gitignore
    |-- README.md

Each project contains its corresponding notebooks, reports,
figures, datasets or processed results.

## Technologies and Methods

- Python
- Jupyter Notebook
- pandas
- scikit-learn
- spaCy
- Matplotlib
- Plotly
- NetworkX
- Statistical signal detection
- Machine learning
- Natural language processing
- Named Entity Recognition
- Relation extraction

## Data Source

FDA Adverse Event Reporting System (FAERS), 2025 Q3.

Official data source:
https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

The original raw FAERS datasets are excluded from this
repository due to their large file sizes. Processed datasets
and analytical results are included where applicable.

## Reproducibility

1. Clone this repository.
2. Download the FAERS 2025 Q3 dataset from the FDA website.
3. Restore the raw datasets to the paths expected by the
   respective notebooks.
4. Install the required Python libraries.
5. Execute the notebooks in sequential order.

For the Mini Project, refer to its README.txt and
results/requirements.txt files for additional information.

## Limitations

FAERS reports are subject to reporting bias, incomplete
information and confounding factors. Statistical associations
do not establish a causal relationship between a drug and
an adverse reaction.

The synthetic text corpus in the Mini Project limits the
generalisability of its NLP evaluation.

This repository is intended for academic and research
demonstration purposes, not clinical decision-making.

## Author

B Aditya

B.Tech - Computational Biology

Academic coursework in AI in Pharmacovigilance.
