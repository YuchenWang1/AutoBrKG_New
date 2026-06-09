## Fact Construction Performance

This method evaluates fact construction performance using precision, recall, and F1 score. The evaluation is conducted on two levels: **triple-level** and **information-level**.

### Overall Metrics

- **Precision:** Calculated by dividing the number of correct triples (those matching the ground truth) by the total number of generated triples.
- **Recall:** Calculated by dividing the number of correct triples by the total number of ground truth triples.
- **F1 Score:** The harmonic mean of precision and recall, providing a comprehensive and reliable evaluation metric, especially for imbalanced data.

### Triple-level Evaluation

Since some Markdown platforms do not support LaTeX formulas, the following formulas are presented in plain text:

- **Precision = TP / (TP + FP)**
- **Recall = TP / (TP + FN)**
- **F1 score = (2 × Precision × Recall) / (Precision + Recall)**

Where:
- **TP:** Number of correctly predicted triples (True Positive)
- **FP:** Number of incorrectly predicted triples (False Positive)
- **FN:** Number of ground truth triples that were not extracted (False Negative)

### Information-level Evaluation

For each category (i.e., subject, object, and relation), the metrics are calculated using the following plain text formulas:

- **Precision_i = TP_i / (TP_i + FP_i)**
- **Recall_i = TP_i / (TP_i + FN_i)**
- **F1 score_i = (2 × Precision_i × Recall_i) / (Precision_i + Recall_i)**

Where:
- **TP_i:** Number of correctly predicted subject, object entities, and relation for the i-th category.
- **FP_i:** Number of incorrectly predicted subject, object entities, and relation for the i-th category.
- **FN_i:** Number of ground truth subject, object entities, and relation for the i-th category that were not extracted.

## Domain Consistency Evaluation

Since general large language models (LLMs) are used instead of domain-specific LLMs, the constructed knowledge graph may include non-domain information. To evaluate domain consistency, the **Hallucination Metric** is employed.

The hallucination metric measures the extent of generated content that is nonsensical or unfaithful to the source information. The following plain text formulas are used:

- **Subject Hallucination (SH) = 1 - (T_sub / T)**
- **Relation Hallucination (RH) = 1 - (T_rel / T)**
- **Object Hallucination (OH) = 1 - (T_obj / T)**

Where:
- **T_sub:** Number of triples with correct subjects.
- **T_rel:** Number of triples with correct relations.
- **T_obj:** Number of triples with correct objects.
- **T:** Total number of generated triples.

All metrics are computed using exact matches in accordance with the domain requirements.
