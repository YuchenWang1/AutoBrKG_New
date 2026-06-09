import argparse
import sys
from pathlib import Path
import json
import re
import jieba
import logging
from typing import List, Dict, Set, Tuple
from collections import Counter

# Configure logging to file for diagnostics
logging.basicConfig(
    filename='evaluation.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def calculate_precision_recall_f1(gold: Set[str], pred: Set[str]) -> Tuple[float, float, float]:
    if not gold:
        return (1.0, 1.0, 1.0) if not pred else (0.0, 0.0, 0.0)
    if not pred:
        return 0.0, 0.0, 0.0
    common = gold & pred
    p = len(common) / len(pred)
    r = len(common) / len(gold)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def calculate_micro_f1_from_lists(gold_list: List[str], sys_list: List[str]) -> Tuple[float, float, float]:
    gold_cnt = Counter(gold_list)
    sys_cnt = Counter(sys_list)
    # True positives: sum of min counts for each triple
    tp = sum(min(sys_cnt[t], gold_cnt[t]) for t in sys_cnt)
    total_sys = sum(sys_cnt.values())
    total_gold = sum(gold_cnt.values())
    p = tp / total_sys if total_sys > 0 else 0.0
    r = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def get_subject_object_hallucinations(
    ontology: Dict,
    test_sentence: str,
    triples: List[List[str]]
) -> Tuple[float, float]:
    if not triples:
        return 0.0, 0.0

    # Raw normalization for substring matching
    raw_text = test_sentence + ''.join(c.get('label', '') for c in ontology.get('concepts', []))
    raw_norm = re.sub(r"\s+", '', raw_text).lower()

    # Token-based matching as fallback
    tokens = set(jieba.cut(test_sentence)) | {
        tok for c in ontology.get('concepts', []) for tok in jieba.cut(c.get('label', ''))
    }
    token_norm = {re.sub(r"\s+", '', tok).lower() for tok in tokens}

    subj_halluc, obj_halluc = 0, 0
    for sub, _, obj in triples:
        norm_sub = clean_entity_string(sub)
        norm_obj = clean_entity_string(obj)
        # if neither raw substring nor token exists, count as hallucination
        if norm_sub not in raw_norm and norm_sub not in token_norm:
            subj_halluc += 1
        if norm_obj not in raw_norm and norm_obj not in token_norm:
            obj_halluc += 1
    total = len(triples)
    return subj_halluc / total, obj_halluc / total


def get_ontology_conformance(
    ontology: Dict,
    triples: List[List[str]]
) -> Tuple[float, float]:
    if not triples:
        return 1.0, 0.0
    ont_rels = {
        rel.get('label', '').strip().replace(' ', '_').lower()
        for rel in ontology.get('relations', [])
    }
    match_count = sum(
        1 for _, rel, _ in triples
        if rel.strip().replace(' ', '_').lower() in ont_rels
    )
    conf = match_count / len(triples)
    return conf, 1.0 - conf


def normalize_triple(sub_label: str, rel_label: str, obj_label: str) -> str:
    sub = re.sub(r"\s+", '', sub_label).lower()
    rel = re.sub(r"\s+", '', rel_label).lower()
    obj = re.sub(r"\s+", '', obj_label).lower()
    return f"{sub}|{rel}|{obj}"


def clean_entity_string(entity: str) -> str:
    joined = ''.join(jieba.cut(entity))
    return re.sub(r"\s+", '', joined).lower()


def read_jsonl(path: Path, is_json: bool = True) -> List:
    data = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line) if is_json else line.strip())
    return data


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_config(config_path: Path) -> Dict:
    cfg = read_json(config_path)
    patterns = cfg.get('path_patterns', {})
    onto_list = []
    for oid in cfg.get('onto_list', []):
        entry = {'id': oid}
        for key, pat in patterns.items():
            entry[key] = Path(pat.replace('$$onto$$', oid))
        onto_list.append(entry)
    return {'onto_list': onto_list, 'avg_out_file': Path(cfg['avg_out_file'])}


def save_jsonl(data: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(item: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_to_dict(data: List[Dict], key: str = 'id') -> Dict[str, Dict]:
    return {str(item[key]): item for item in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_config_path', type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.eval_config_path)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    cfg = load_config(config_path)
    avg_out = cfg['avg_out_file']
    if avg_out.exists():
        avg_out.unlink()

    # Accumulators for macro (sentence-level) and micro (triple-level) global metrics
    macro_totals = {'p': 0.0, 'r': 0.0, 'f1': 0.0, 'conf': 0.0, 'sub_h': 0.0, 'rel_h': 0.0, 'obj_h': 0.0}
    macro_count = 0

    global_flat_gold_list: List[str] = []
    global_flat_sys_list: List[str] = []

    for onto in cfg['onto_list']:
        oid = onto['id']
        logging.info(f"Processing ontology: {oid}")

        sys_data = convert_to_dict(read_jsonl(onto['sys']))
        gt_data = convert_to_dict(read_jsonl(onto['gt']))
        onto_json = read_json(onto['onto'])

        per_sent_metrics = []
        sums = {k: 0.0 for k in macro_totals}
        n_cases = len(gt_data)
        if n_cases == 0:
            logging.warning(f"No test cases for {oid}, skipping.")
            continue

        onto_flat_gold_list: List[str] = []
        onto_flat_sys_list: List[str] = []

        for sid, gt in gt_data.items():
            sent = gt.get('sent', '')
            gt_triples = [[str(t['sub']), str(t['rel']), str(t['obj'])] for t in gt.get('triples', [])]

            raw_sys = sys_data.get(sid, {}).get('triples', [])
            sys_triples: List[List[str]] = []
            for t in raw_sys:
                if isinstance(t, list) and len(t) == 3:
                    sys_triples.append([str(e or '') for e in t])
                elif isinstance(t, dict) and all(k in t for k in ('sub', 'rel', 'obj')):
                    sys_triples.append([str(t['sub'] or ''), str(t['rel'] or ''), str(t['obj'] or '')])

            # accumulate flat lists (with duplicates)
            for sub, rel, obj in gt_triples:
                onto_flat_gold_list.append(normalize_triple(sub, rel, obj))
                global_flat_gold_list.append(normalize_triple(sub, rel, obj))
            for sub, rel, obj in sys_triples:
                onto_flat_sys_list.append(normalize_triple(sub, rel, obj))
                global_flat_sys_list.append(normalize_triple(sub, rel, obj))

            gt_relset = {r.strip().replace(' ', '_').lower() for _, r, _ in gt_triples}
            filtered = [t for t in sys_triples if t[1].strip().replace(' ', '_').lower() in gt_relset]

            norm_gt = {normalize_triple(*t) for t in gt_triples}
            norm_sys = {normalize_triple(*t) for t in filtered}

            p, r, f1 = calculate_precision_recall_f1(norm_gt, norm_sys)
            conf, rel_h = get_ontology_conformance(onto_json, sys_triples)
            sub_h, obj_h = get_subject_object_hallucinations(onto_json, sent, sys_triples)

            if f1 < 1.0 and filtered and sub_h == 0 and obj_h == 0:
                logging.info(f"ID {sid}: sent='{sent}' f1={f1:.4f} sys={filtered} gt={gt_triples}")

            metrics = {
                'id': sid,
                'precision': round(p, 4),
                'recall': round(r, 4),
                'f1': round(f1, 4),
                'onto_conf': round(conf, 5),
                'rel_halluc': round(rel_h, 5),
                'sub_halluc': round(sub_h, 5),
                'obj_halluc': round(obj_h, 5),
                'llm_triples': sys_triples,
                'filtered_llm_triples': filtered,
                'gt_triples': gt_triples,
                'sent': sent
            }
            per_sent_metrics.append(metrics)
            for k, v in zip(sums.keys(), (p, r, f1, conf, sub_h, rel_h, obj_h)):
                sums[k] += v

        save_jsonl(per_sent_metrics, onto['output'])

        # Sentence-level macro averages
        avg = {k: sums[k] / n_cases for k in sums}
        summary = {
            'onto': oid,
            'type': 'all_test_cases',
            'avg_precision': round(avg['p'], 5),
            'avg_recall': round(avg['r'], 5),
            'avg_f1': round(avg['f1'], 5),
            'avg_onto_conf': round(avg['conf'], 5),
            'avg_sub_halluc': round(avg['sub_h'], 5),
            'avg_rel_halluc': round(avg['rel_h'], 5),
            'avg_obj_halluc': round(avg['obj_h'], 5)
        }
        append_jsonl(summary, avg_out)

        # Flat-triples micro-level for this ontology
        micro_p, micro_r, micro_f1 = calculate_micro_f1_from_lists(
            onto_flat_gold_list, onto_flat_sys_list
        )
        flat_summary = {
            'onto': oid,
            'type': 'flat_triples',
            'precision': round(micro_p, 5),
            'recall': round(micro_r, 5),
            'f1': round(micro_f1, 5)
        }
        append_jsonl(flat_summary, avg_out)

        # accumulate for global macro
        for k in macro_totals:
            macro_totals[k] += avg['p' if k == 'p' else k]
        macro_count += 1

    # Global summaries
    if macro_count > 0:
        # Global macro
        final_macro = {k: macro_totals[k] / macro_count for k in macro_totals}
        global_summary = {
            'id': 'global_summary',
            'type': 'average_over_ontologies',
            'num_ontologies_processed': macro_count,
            'avg_precision': round(final_macro['p'], 5),
            'avg_recall': round(final_macro['r'], 5),
            'avg_f1': round(final_macro['f1'], 5),
            'avg_onto_conf': round(final_macro['conf'], 5),
            'avg_sub_halluc': round(final_macro['sub_h'], 5),
            'avg_rel_halluc': round(final_macro['rel_h'], 5),
            'avg_obj_halluc': round(final_macro['obj_h'], 5)
        }
        append_jsonl(global_summary, avg_out)

        # Global flat-triples micro (multiset)
        g_p, g_r, g_f1 = calculate_micro_f1_from_lists(
            global_flat_gold_list, global_flat_sys_list
        )
        global_flat = {
            'id': 'global_flat_triples',
            'type': 'flat_triples',
            'precision': round(g_p, 5),
            'recall': round(g_r, 5),
            'f1': round(g_f1, 5)
        }
        append_jsonl(global_flat, avg_out)

        print(f"Global summaries appended to {avg_out}")
    else:
        print("No ontologies processed. Global summary skipped.")

    print("Evaluation complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
