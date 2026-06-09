import json
import os
import re
from typing import List, Dict, Set, Tuple


def calculate_stats(gold: Set[str], pred: Set[str]) -> Dict[str, float]:
    """
    TP / FP / FN & P / R / F1
    """
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def normalize_triple(sub_label: str, rel_label: str, obj_label: str) -> str:

    sub_norm = re.sub(r"\s+", '', sub_label).lower()
    rel_norm = re.sub(r"\s+", '', rel_label).lower()
    obj_norm = re.sub(r"\s+", '', obj_label).lower()
    return f"{sub_norm}_{rel_norm}_{obj_norm}"


def read_jsonl(jsonl_path: str) -> List[Dict]:
    data = []
    with open(jsonl_path, encoding='utf-8') as in_file:
        for line in in_file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} for line: {line.strip()}")
    return data


def convert_to_dict(data: List[Dict], id_name: str = "id") -> Dict[str, Dict]:
    return {item[id_name]: item for item in data}


def evaluate_entities_and_relationships(
    ground_truth: List[Dict], system_output: List[Dict]
) -> Dict[str, Dict[str, float]]:
    entity_sets: Dict[str, Dict[str, Set[str]]] = {}
    relation_sets: Dict[str, Dict[str, Set[str]]] = {}

    # GOLD
    for item in ground_truth:
        for triple in item.get('triples', []):
            sub = triple['sub']['name']
            obj = triple['obj']['name']
            et_sub = triple['sub']['type']
            et_obj = triple['obj']['type']
            raw_rel = triple['rel']['name']
            rel_norm = re.sub(r"\s+", '', raw_rel).lower()

            for et, ent in ((et_sub, sub), (et_obj, obj)):
                entity_sets.setdefault(et, {'gold': set(), 'pred': set()})
                entity_sets[et]['gold'].add(ent)

            relation_sets.setdefault(rel_norm, {'gold': set(), 'pred': set()})
            relation_sets[rel_norm]['gold'].add(normalize_triple(sub, raw_rel, obj))

    # PRED
    for item in system_output:
        for triple in item.get('triples', []):
            sub = triple['sub']['name']
            obj = triple['obj']['name']
            et_sub = triple['sub']['type']
            et_obj = triple['obj']['type']
            raw_rel = triple['rel']['name']
            rel_norm = re.sub(r"\s+", '', raw_rel).lower()

            if et_sub in entity_sets:
                entity_sets[et_sub]['pred'].add(sub)
            if et_obj in entity_sets:
                entity_sets[et_obj]['pred'].add(obj)

            if rel_norm in relation_sets:
                relation_sets[rel_norm]['pred'].add(normalize_triple(sub, raw_rel, obj))

    # metrics
    evaluation_results: Dict[str, Dict[str, float]] = {}

    for et, sets in entity_sets.items():
        evaluation_results[f"entity_{et}"] = calculate_stats(sets['gold'], sets['pred'])

    for rel, sets in relation_sets.items():
        evaluation_results[f"relation_{rel}"] = calculate_stats(sets['gold'], sets['pred'])

    return evaluation_results

def main():
    config_path = "config/bridge.jsonl"
    output_path = "bridge-class_results.txt"

    if not os.path.exists(config_path):
        print(f"Evaluation config file not found: {config_path}")
        return

    eval_inputs = read_jsonl(config_path)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for onto in eval_inputs:
            gt_dict = convert_to_dict(read_jsonl(onto['gt']))
            sys_dict = convert_to_dict(read_jsonl(onto['sys']))

            results = evaluate_entities_and_relationships(
                ground_truth=list(gt_dict.values()),
                system_output=list(sys_dict.values())
            )

            out_f.write(f"Evaluation results for {onto['id']}:\n")
            for key, metrics in results.items():
                if key.startswith('entity_'):
                    tag = "Entity"
                    name = key[len('entity_'):]
                else:
                    tag = "Relation"
                    name = key[len('relation_'):]

                out_f.write(
                    f"{tag}: {name:<20} | "
                    f"TP: {metrics['tp']:<4} FP: {metrics['fp']:<4} FN: {metrics['fn']:<4} | "
                    f"P: {metrics['precision']:.4f} R: {metrics['recall']:.4f} F1: {metrics['f1']:.4f}\n"
                )
            out_f.write("\n")


if __name__ == "__main__":
    main()
