# Baselines

This directory contains the reproducibility package for the comparison baselines used in the paper.

## Compared Methods

- Gemini 2.5 Pro Preview 05-06
- Qwen 3.6 Plus Preview
- DeepSeek V3
- Llama-3.2-3B-Instruct
- LightRAG
- KGGEn / KGGen
- EDC

## Directory Layout

- `prompts/llm_bridge_kg_extraction_prompt.md`: shared prompt for direct LLM extraction baselines.
- `model_matrix.json`: canonical baseline names, model IDs, and prompt mapping.
- `third_party_sources.md`: official source links for framework baselines that should not be vendored here.
- `EDC_settings/`: project-specific EDC prompt templates, bridge schema, and few-shot files copied from the experiment folder.

The goal is to keep this repository clear and redistributable while preserving enough detail to reproduce the comparison setup.
