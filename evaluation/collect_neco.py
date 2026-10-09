"""Collect all scheduled NeCo tasks without selecting favourable seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def collect(root):
    tasks = json.loads((root / 'tasks.json').read_text())
    groups = defaultdict(list)
    missing = []
    splits = defaultdict(list)
    for index, task in enumerate(tasks):
        out = root / f'task_{index:03d}_{task["evaluation"]}_{task["dataset"]}_{task["model"]}'
        key = (task['evaluation'], task['dataset'], task['model'], task.get('fraction'), task.get('clusters'))
        result_file = out / 'results.json'
        if not result_file.exists():
            missing.append(index)
            groups[key].append(None)
            continue
        result = json.loads(result_file.read_text())
        assert result['complete'] and result['task'] == task
        score = (result['miou_percent'] if task['evaluation'] == 'retrieval'
                 else result['final_miou_percent_448px'] if task['evaluation'] == 'linear'
                 else result['community_detection_mean_miou_percent'] if task['evaluation'] == 'fully_unsupervised'
                 else result['mean_miou_percent'])
        groups[key].append(float(score))
        manifest = out / 'split_manifest.json'
        if manifest.exists():
            splits[(task['evaluation'], task['dataset'])].append(json.loads(manifest.read_text()))
    for key, manifests in splits.items():
        assert all(x == manifests[0] for x in manifests), f'Models used different COCO splits: {key}'
    report = []
    for key, values in groups.items():
        available = [v for v in values if v is not None]
        complete = len(available) == len(values)
        report.append({'evaluation': key[0], 'dataset': key[1], 'model': key[2],
                       'fraction': key[3], 'clusters': key[4], 'complete': complete,
                       'expected_tasks': len(values), 'completed_tasks': len(available),
                       'mean_miou_percent': statistics.mean(available) if complete else None,
                       'seed_std_percent': statistics.pstdev(available) if complete else None,
                       'individual_scores_percent': available})
    result = {'complete': not missing, 'completed_tasks': len(tasks) - len(missing),
              'expected_tasks': len(tasks), 'missing_task_indices': missing,
              'groups': report, 'completed_coco_model_split_comparisons_verified': True}
    tmp = root / 'results_summary.json.tmp'
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(root / 'results_summary.json')
    print(f'NeCo tasks complete: {len(tasks) - len(missing)}/{len(tasks)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    collect(parser.parse_args().root)
