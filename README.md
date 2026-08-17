# Demographic Bias and Bias Mitigation in Toxic Comment Classification

## Overview

This project investigates **demographic bias in toxic-comment classification** using machine learning and BERT. It compares three classification approaches—**Logistic Regression, Random Forest, and BERT**—and evaluates both predictive performance and demographic fairness.

The project then tests three bias-mitigation strategies:

1. **Pre-processing:** statistical reweighting
2. **In-processing:** Fairlearn Exponentiated Gradient with a Demographic Parity constraint
3. **Post-processing:** validation-selected group-specific decision thresholds

The overall objective is to determine whether toxicity classifiers can maintain useful predictive performance while reducing disparities across demographic identity-related subgroups.

> **Important dataset note:** The project report discusses CivilComments-WILDS, and the notebook actually loads `pietrolesci/civilcomments-wilds` with the `default` configuration. The Google Civil Comments dataset supplied as the project dataset link is the broader/original Civil Comments resource. The WILDS-derived version is used because it provides the demographic identity indicators required for the fairness experiments.

## Research Question

> To what extent do machine-learning models for toxic-comment detection exhibit demographic bias, and which bias-mitigation techniques are most effective at improving fairness without compromising classification performance?

## Objectives

- Develop and evaluate Logistic Regression, Random Forest, and fine-tuned BERT models for binary toxic-comment classification.
- Assess demographic bias using Demographic Parity Difference, Equal Opportunity Difference, and False Positive Rate Difference.
- Compare pre-processing, in-processing, and post-processing mitigation methods and evaluate their fairness-performance trade-offs.

## Dataset

### Dataset used in the notebook

**CivilComments-WILDS**

- Hugging Face dataset: `pietrolesci/civilcomments-wilds`
- Configuration: `default`
- Text column: `comment_text`
- Target column: `toxicity`
- Dataset splits: `train`, `validation`, `test`
- Total records used by the project dataset: approximately 445,293
- Toxicity is treated as a binary classification target.

The dataset provides eight identity-related indicators:

- `male`
- `female`
- `LGBTQ`
- `christian`
- `muslim`
- `other_religions`
- `black`
- `white`

An additional `identity_any` indicator is used for controlled mitigation experiments.

### Original dataset reference

Google Civil Comments is available on Hugging Face:

https://huggingface.co/datasets/google/civil_comments

The Google dataset contains the original Civil Comments toxicity-related labels, while the WILDS-derived dataset used by this implementation supplies the demographic indicators needed for the fairness analysis.

## Methodology

The notebook is organised as a four-week experimental pipeline.

### Week 1 — Dataset Validation, EDA and Leakage Control

- Load the verified CivilComments-WILDS dataset.
- Validate the expected `train`, `validation`, and `test` partitions.
- Convert data to pandas for controlled experimentation.
- Normalise whitespace and handle missing text.
- Check duplicates and potential cross-split leakage.
- Generate SHA-256 hashes of normalised comments for leakage checking.
- Preserve official dataset partitions.
- Create reproducible stratified samples using random seed `42`.

### Week 2 — Classical Models and Fairness Audit

#### TF-IDF

The classical models use TF-IDF representations with:

- Unigrams and bigrams
- Maximum 25,000 features
- Minimum document frequency of 3
- Sublinear TF scaling
- Unicode accent handling

#### Logistic Regression

- `C = 2.0`
- `liblinear` solver
- `max_iter = 1000`
- Balanced class weights
- Random seed = 42

#### Random Forest

- 180 trees
- Maximum depth = 35
- Minimum leaf size = 2
- `sqrt` feature selection
- Balanced subsample weighting
- Parallel training

### Week 3 — BERT and Bias Mitigation

#### BERT

The transformer model is:

`bert-base-uncased`

Configuration:

- Maximum sequence length: 128 tokens
- Batch size: 16
- Epochs: 2
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Warm-up ratio: `0.10`
- Mixed-precision training
- NVIDIA T4 GPU recommended/required for the notebook's BERT stage

Because the project was developed within free Google Colab constraints, BERT was trained using a controlled subset rather than the entire corpus.

#### Pre-processing mitigation — Reweighting

Training observations are assigned weights according to the relationship between:

- demographic/identity status
- toxicity label

The method increases the contribution of under-represented combinations and reduces the contribution of over-represented combinations.

#### In-processing mitigation — Exponentiated Gradient

Fairlearn's `ExponentiatedGradient` is used with:

- Logistic Regression as the base estimator
- Demographic Parity constraint
- Difference bound = `0.05`
- `eps = 0.05`
- Maximum iterations = 20

#### Post-processing mitigation — Threshold Adjustment

Group-specific decision thresholds are selected using the validation set.

- Threshold grid: `0.25` to `0.75`
- Step size: `0.05`
- The test set is not used for threshold selection.
- The selected thresholds are then applied to the held-out test set.

### Week 4 — Final Comparison and Explainability

The final stage includes:

- Model comparison
- Confusion matrices
- Fairness comparison tables
- Logistic Regression feature coefficients
- SHAP analysis
- False-positive and false-negative error analysis
- Automated evidence generation
- Final experimental summary

## Fairness Metrics

The project uses three complementary fairness measures.

### Demographic Parity Difference (DPD)

Measures the difference in positive prediction rates between an identity subgroup and the reference population.

Values closer to zero indicate smaller prediction-rate disparities.

### Equal Opportunity Difference (EOD)

Measures the difference in true-positive rates between groups.

This is important for determining whether genuinely toxic comments are detected with similar sensitivity across groups.

### False Positive Rate Difference (FPRD)

Measures differences in false-positive rates.

This is particularly important for toxicity moderation because a high false-positive disparity can cause legitimate comments involving demographic identities to be incorrectly flagged.

> No single fairness metric represents fairness in every sense. The project therefore evaluates multiple metrics alongside classification performance.

## Experimental Sampling

To make the experiments feasible in Google Colab, controlled samples are used:

| Experiment | Maximum records |
|---|---:|
| Classical model training | 100,000 |
| Random Forest training | 50,000 |
| Common validation set | 8,000 |
| Common test set | 10,000 |
| BERT training | 20,000 |
| BERT validation | 4,000 |
| Mitigation training | 25,000 |

The same common test set is used for the principal Logistic Regression, Random Forest, and BERT comparison.

## Results Summary

The final report found that **BERT provided the strongest overall predictive performance**, while fairness results varied by metric and model.

### Classification performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.87 | 0.45 | 0.72 | 0.55 | 0.89 |
| Random Forest | 0.89 | 0.53 | 0.41 | 0.46 | 0.82 |
| BERT | 0.9191 | — | 0.5928 | 0.6249 | 0.9280 |

BERT achieved the highest F1 score and ROC-AUC. Logistic Regression achieved higher recall, but with substantially lower precision.

### Baseline fairness

The baseline models showed measurable demographic disparities.

- Logistic Regression produced the largest Demographic Parity Difference and False Positive Rate Difference.
- Random Forest reduced some disparities but still showed substantial subgroup gaps.
- BERT produced lower maximum Demographic Parity and False Positive Rate disparities than the classical baselines.
- No baseline model was consistently fairest across every fairness metric.

### Mitigation results

The mitigation experiments demonstrated a clear fairness-performance trade-off.

**Pre-processing reweighting** substantially reduced demographic disparities, but toxic-comment recall dropped sharply.

**Exponentiated Gradient** achieved strong reductions in selected fairness disparities, but also produced a severe reduction in recall and F1.

**Post-processing thresholds** preserved considerably more predictive utility, but did not improve every fairness metric simultaneously.

The main conclusion is therefore that **a lower fairness disparity does not automatically mean a better operational classifier**. Fairness metrics must be interpreted together with recall, precision, F1, and the practical consequences of false positives and false negatives.

## Explainability

The project uses:

- Logistic Regression coefficients
- TF-IDF feature importance
- SHAP analysis
- High-confidence false-positive analysis
- High-confidence false-negative analysis

The Logistic Regression analysis identified strongly predictive abusive terms, while some identity-related terms also appeared among important toxicity predictors. This provides evidence for investigating possible identity-term associations rather than assuming that the model is making decisions solely from explicit abusive language.

SHAP is used as a complementary interpretability method; feature importance should not automatically be interpreted as evidence that every highly ranked feature increases toxicity.

## Project Structure

A recommended repository structure is:

```text
.
├── README.md
├── Chandu_CivilComments_Fairness_Complete_4_Weeks_Colab.ipynb
├── results/
│   ├── tables/
│   ├── figures/
│   └── report_evidence/
└── requirements.txt
```

The notebook itself automatically creates the following working directories in Google Colab:

```text
/content/civilcomments_fairness_results/
├── tables/
├── figures/
├── report_evidence/
└── temporary_models/
```

The notebook also generates an `experiment_config.json` file and a machine-readable `final_experimental_summary.md`.

## Installation

The notebook installs the main dependencies automatically.

Core packages include:

```bash
pip install "datasets>=3.2,<5" \
            "transformers>=4.48,<5" \
            "accelerate>=1.2,<2" \
            "fairlearn>=0.12,<0.14" \
            "shap>=0.46,<0.50"
```

The project also uses:

- Python
- NumPy
- pandas
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Fairlearn
- SHAP
- Matplotlib

## How to Run

### Option 1 — Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Install the required packages using the first notebook cell.
3. Use a **T4 GPU** runtime for the BERT section.
4. Run the notebook cells sequentially.
5. Allow the dataset to download from Hugging Face.
6. Review the generated tables, figures and report evidence.
7. Check the final experimental summary generated near the end of the notebook.

### Option 2 — Local environment

A CUDA-enabled environment is recommended if you want to reproduce the BERT experiments locally.

After installing the dependencies, open the notebook and execute the cells in order.

## Reproducibility

The project uses:

```text
Random seed: 42
```

The seed is applied to Python, NumPy and PyTorch where applicable.

Reproducibility is further supported by:

- fixed sampling procedures
- fixed model configurations
- shared test data
- explicit validation/test separation
- controlled threshold selection
- leakage checks
- automated evidence generation

## Important Limitations

This is a research prototype rather than a production content-moderation system.

Key limitations include:

- English-language comments only.
- Demographic indicators represent identity references in comments, not verified author demographics.
- Only eight general demographic indicators are evaluated.
- BERT is trained on a controlled subset because of Google Colab computational limits.
- Fairness metrics can conflict with one another.
- Some mitigation strategies substantially reduce recall.
- The experiments do not cover multilingual moderation.
- No real-time moderation deployment is included.
- No moderator escalation or appeals workflow is implemented.
- Results should not be interpreted as evidence that every subgroup disparity represents intentional discrimination.

## Ethical Considerations

Toxic-comment classification can affect freedom of expression and participation in online communities. False positives may suppress legitimate discussions, especially when comments mention demographic identities. False negatives may allow harmful content to remain visible.

For this reason, the project treats fairness as a multi-dimensional evaluation problem rather than a single optimisation target.

Any practical deployment should include:

- Human oversight
- Regular fairness audits
- Transparent performance reporting
- Error monitoring
- Appropriate review and appeals processes
- Evaluation across relevant demographic and linguistic populations

## Key References

- Borkan, D., Dixon, L., Sorensen, J., Thain, N. & Vasserman, L. (2019). *Nuanced metrics for measuring unintended bias with real data for text classification.*
- Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Dixon, L., Li, J., Sorensen, J., Thain, N. & Vasserman, L. (2018). *Measuring and Mitigating Unintended Bias in Text Classification.*
- Hardt, M., Price, E. & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.*
- Koh, P. W. et al. (2021). *WILDS: A Benchmark of in-the-Wild Distribution Shifts.*
- Suresh, H. & Guttag, J. (2021). *A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle.*

## Citation and Dataset Attribution

If you use the Civil Comments dataset, cite and acknowledge the original dataset and the dataset version used for the experiments.

Original dataset:
https://huggingface.co/datasets/google/civil_comments

WILDS-derived dataset used by the notebook:
https://huggingface.co/datasets/pietrolesci/civilcomments-wilds

This repository should not redistribute the dataset itself. The notebook downloads the dataset from its Hugging Face source.

## Author

**Purna Chandu Vangala**

MSc Data Science  
7005SCN Individual Research Project  
Academic Year 2025/26

Project title:

**Demographic Bias and Bias Mitigation in Toxic Comment Classification Using Machine Learning and BERT**
