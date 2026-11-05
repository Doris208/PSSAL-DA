import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate external-method efficiency benchmark results.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def load_results(input_dir):
    results = []
    for path in Path(input_dir).glob('*_efficiency.json'):
        with open(path, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    return results


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        'Method', 'Task', 'Backbone', 'Input', 'Checkpoint Loaded',
        'Active Parameters(M)', 'Student Total Parameters(M)', 'GFLOPs',
        'GMACs', 'FPS Mean', 'FPS Std', 'Latency Mean(ms)',
        'Latency P95(ms)', 'Peak Memory(MB)', 'GPU', 'Precision', 'Status',
        'Error Type', 'Error Message'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        'Method', 'Backbone', 'Parameters(M)', 'Complexity(G)', 'FPS',
        'architecture_mismatch', 'status', 'error_messages', 'tasks'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_rows(results):
    rows = []
    for r in results:
        rows.append({
            'Method': r.get('method'),
            'Task': r.get('task'),
            'Backbone': f"{r.get('backbone_type')}-{r.get('backbone_depth')}",
            'Input': 'Image+NDVI+NDRE+CIRE',
            'Checkpoint Loaded': r.get('checkpoint_loaded'),
            'Active Parameters(M)': r.get('active_inference_parameters_m'),
            'Student Total Parameters(M)': r.get('student_total_parameters_m'),
            'GFLOPs': r.get('gflops'),
            'GMACs': r.get('gmacs'),
            'FPS Mean': r.get('fps_mean'),
            'FPS Std': r.get('fps_std'),
            'Latency Mean(ms)': r.get('latency_mean_ms'),
            'Latency P95(ms)': r.get('latency_p95_ms'),
            'Peak Memory(MB)': r.get('peak_memory_mb'),
            'GPU': r.get('gpu_name'),
            'Precision': r.get('precision'),
            'Status': r.get('status'),
            'Error Type': r.get('error_type'),
            'Error Message': r.get('error_message'),
        })
    return rows


def summarize(results):
    groups = defaultdict(list)
    for result in results:
        groups[result.get('method')].append(result)
    summary = []
    for method, items in groups.items():
        successful = [
            i for i in items
            if i.get('status') in ('OK', 'DRY_RUN_OK')
            and i.get('active_inference_parameters_m') is not None
            and i.get('gflops') is not None
        ]
        if not successful:
            summary.append({
                'Method': method,
                'Backbone': f"{items[0].get('backbone_type')}-{items[0].get('backbone_depth')}",
                'Parameters(M)': None,
                'Complexity(G)': None,
                'FPS': None,
                'architecture_mismatch': None,
                'status': 'FAILED',
                'error_messages': [
                    i.get('error_message') for i in items
                    if i.get('error_message')
                ],
                'tasks': [i.get('task') for i in items],
            })
            continue
        items = successful
        params = [i.get('active_inference_parameters_m') for i in items]
        gflops = [i.get('gflops') for i in items]
        fps = [i.get('fps_mean') for i in items if i.get('fps_mean') is not None]
        arch_mismatch = len({round(p, 6) for p in params if p is not None}) > 1
        gf_ok = len([g for g in gflops if g is not None]) == len(items)
        if gf_ok and gflops:
            base = gflops[0]
            arch_mismatch = arch_mismatch or any(
                abs(g - base) / max(abs(base), 1e-9) > 0.001 for g in gflops)
        summary.append({
            'Method': method,
            'Backbone': f"{items[0].get('backbone_type')}-{items[0].get('backbone_depth')}",
            'Parameters(M)': None if arch_mismatch else params[0],
            'Complexity(G)': None if arch_mismatch else gflops[0],
            'FPS': sum(fps) / len(fps) if fps else None,
            'architecture_mismatch': arch_mismatch,
            'status': 'OK' if len(successful) == len(groups[method])
            else 'PARTIAL',
            'error_messages': [
                i.get('error_message') for i in groups[method]
                if i.get('error_message')
            ],
            'tasks': [i.get('task') for i in items],
        })
    return summary


def write_markdown(path, summary):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('| Method | Backbone | Parameters(M) | Complexity(G) | FPS |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for row in summary:
            f.write(
                f"| {row['Method']} | {row['Backbone']} | "
                f"{row['Parameters(M)']} | {row['Complexity(G)']} | {row['FPS']} |\n")


def write_latex(path, summary):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{lrrr}\n')
        f.write('\\toprule\nMethod & Parameters(M) & Complexity(G) & FPS \\\\\n')
        f.write('\\midrule\n')
        for row in summary:
            f.write(
                f"{row['Method']} & {row['Parameters(M)']} & "
                f"{row['Complexity(G)']} & {row['FPS']} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.input_dir)
    rows = raw_rows(results)
    summary = summarize(results)
    write_csv(os.path.join(args.output_dir,
                           'external_methods_efficiency_raw.csv'), rows)
    write_summary_csv(os.path.join(
        args.output_dir, 'external_methods_efficiency_summary.csv'), summary)
    with open(os.path.join(args.output_dir,
                           'external_methods_efficiency_summary.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    write_markdown(os.path.join(args.output_dir,
                                'external_methods_efficiency_summary.md'),
                   summary)
    write_latex(os.path.join(args.output_dir,
                             'external_methods_efficiency_summary.tex'),
                summary)
    print(f'Aggregated {len(results)} result files into {args.output_dir}')


if __name__ == '__main__':
    main()
