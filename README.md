# Superpixel Correlation for Explainable Image Classification

This repository is the official implementation of [Superpixel Correlation for Explainable Image Classification](). 

In this paper, we introduce Correlation SHAP (CorrSHAP), a novel approach that leverages image superpixel correlations to significantly accelerate SHAP value estimation while preserving the rigor of the original formulation. 

![Method_Framewrok](image.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Results

### Qualitative evaluation

Example of correlations for different proposed approaches for random ImageNet image. Run concept_importance.ipynb and Evaluation_ConceptCorrelationSHAP.ipynb to reproduce these results.

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

![Qualitative evaluation](image-1.png)

### Quantitative results

Running concept_correlation_SHAP.py

![alt text](image-6.png)