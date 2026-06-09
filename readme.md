# AutoBrKG

AutoBrKG constructs domain knowledge graphs from infrastructure inspection reports with a multi-agent extraction, validation, correction, memory retrieval, and graph-construction pipeline.

## Key Policy for Open Review

- `max_iterations` is set to `3` for the validation-correction loop.
- The dynamic memory repository `L` is updated only when the final Validator score is `>= 1.0`.
- Outputs with `score < 1.0` are not admitted to `L` and are not added to the final graph. They are routed to a low-confidence/manual-review queue.
- The memory retrieval module uses validated examples to guide few-shot prompting without accepting unverified low-confidence candidates.

## Repository Layout

- `agents/`: decomposer, extractor, validator, corrector, constructor, and reviewer agents.
- `knowledge/`: bridge-domain retrieval memory and RAG utilities.
- `utils1/`: ontology validation and Turtle conversion utilities.
- `evaluation/`: scripts and data formats for triple-level and ontology-level evaluation.
- `cross_domain/`: road and tunnel prompts, memory bases, ontologies, and 50-record source samples.
- `baselines/`: baseline prompt settings, model list, and third-party source notes.
- `data/inspection_report.txt`: small bridge input example.

## Running the Bridge Pipeline

1. Configure chat, embedding, and Neo4j credentials through environment variables referenced in `configs/model_config.json`.
2. Place the inspection report in `data/inspection_report.txt`.
3. Run:

```bash
export DEEPSEEK_API_KEY="..."
export ZHIPUAI_API_KEY="..."
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="..."
python main.py
```

## Evaluation

Triple-level and ontology-level evaluation scripts are under `evaluation/src/`. See `evaluation/mertic_information.md` for metric definitions.

## Notes

Chinese and English prompts are intentionally preserved because the experiments use Chinese inspection reports with English task guards. Code comments and documentation are written primarily in English for open-source review.
