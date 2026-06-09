# Cross-Domain Materials

This directory packages the road and tunnel resources used to extend AutoBrKG beyond bridge inspection reports.

## Contents

- `road/` contains road-domain prompts, memory-retrieval knowledge bases, ontology files, and 50 sampled source records.
- `tunnel/` contains tunnel-domain prompts, memory-retrieval knowledge bases, ontology files, and 50 sampled source records.

The prompts keep the original Chinese extraction instructions and the English system/task guards used in the experiments. The surrounding documentation and code comments are written in English for open-source review.

## Sampling Policy

The `source_sample_50` files were selected from each domain's `data/inspection_report.txt` with random seed `20260609`. The sampled records are intended as representative open-source examples; full raw experimental corpora can remain outside the public repository if licensing or privacy review requires it.

## Memory Retrieval

The knowledge-base files are used as retrieval memory for few-shot examples. New examples are admitted to the memory repository `L` only when the Validator score is `>= 1.0`; lower-scored candidates must go to the manual-review queue.
