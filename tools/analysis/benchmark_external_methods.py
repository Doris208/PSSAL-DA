import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from model_benchmark_adapter import build_adapter
from model_efficiency_utils import (benchmark_fps, collect_active_parameters,
                                    count_flops_hook, count_parameters,
                                    get_environment_info, make_img_metas,
                                    save_json)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark external-method deployment efficiency.')
    parser.add_argument('--manifest')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--method-name')
    parser.add_argument('--task-name')
    parser.add_argument('--adapter', default='default')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-shape', nargs=3, type=int, default=[6, 256, 256])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--precision', default='fp32')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output-dir',
                        default='/root/shared-nvme/benchmark_results/external_methods')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def checkpoint_from_entry(entry):
    env_name = entry.get('checkpoint_env')
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    return entry.get('checkpoint')


def entries_from_args(args):
    if args.manifest:
        manifest = load_manifest(args.manifest)
        defaults = manifest.get('benchmark_defaults', {})
        entries = manifest['methods']
        return defaults, entries
    return {}, [
        dict(
            name=args.method_name,
            task=args.task_name,
            config=args.config,
            checkpoint=args.checkpoint,
            adapter=args.adapter)
    ]


def structure_info(cfg):
    model = cfg.model
    backbone = model.get('backbone', {})
    decode_head = model.get('decode_head', {})
    return dict(
        model_type=model.get('type'),
        backbone_type=backbone.get('type'),
        backbone_depth=backbone.get('depth'),
        backbone_in_channels=backbone.get('in_channels'),
        decode_head_type=decode_head.get('type'),
        decode_head_num_classes=decode_head.get('num_classes'),
        test_cfg=dict(model.get('test_cfg', {})))


def benchmark_one(entry, args, defaults):
    method = entry['name']
    task = entry['task']
    config = entry['config']
    checkpoint = checkpoint_from_entry(entry)
    adapter_name = entry.get('adapter', 'default')
    output_name = f'{method}_{task}_efficiency.json'.replace(' ', '_').replace('/', '_')
    output_path = os.path.join(args.output_dir, output_name)
    result = dict(
        method=method,
        task=task,
        config=config,
        checkpoint=checkpoint,
        checkpoint_loaded=False,
        adapter=adapter_name,
        status='FAILED')
    try:
        device = torch.device(args.device if torch.cuda.is_available()
                              and args.device.startswith('cuda') else 'cpu')
        adapter = build_adapter(adapter_name)
        model, cfg, adapter_info = adapter.build_deploy_model(
            config, checkpoint, device=device)
        checkpoint_info = adapter_info['checkpoint_info']
        result.update(checkpoint_info)
        result.update(adapter_info)
        result.update(structure_info(cfg))

        c, h, w = args.input_shape
        dummy = torch.randn(args.batch_size, c, h, w, device=device)
        img_metas = make_img_metas(h, w, c)
        active_params, active_modules = collect_active_parameters(
            model, adapter.forward_deploy_logits, dummy, img_metas)
        full_params = count_parameters(model)
        flop_info = count_flops_hook(
            model, adapter.forward_deploy_logits, dummy, img_metas)

        result.update(
            full_object_parameters_m=full_params / 1e6,
            student_total_parameters_m=full_params / 1e6,
            active_inference_parameters_m=active_params / 1e6,
            active_module_types=active_modules,
            input_shape=[args.batch_size, c, h, w],
            precision=args.precision,
            batch_size=args.batch_size)
        result.update(flop_info)
        if args.dry_run:
            result.update(
                fps_mean=None,
                fps_std=None,
                fps_min=None,
                fps_max=None,
                latency_mean_ms=None,
                latency_p50_ms=None,
                latency_p95_ms=None,
                peak_memory_mb=None,
                status='DRY_RUN_OK')
        else:
            result.update(
                benchmark_fps(
                    model,
                    adapter.forward_deploy_logits,
                    dummy,
                    img_metas,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeat=args.repeat))
        result.update(get_environment_info(device))
    except Exception as exc:
        result.update(
            status='FAILED',
            error_type=type(exc).__name__,
            error_message=str(exc))
    save_json(output_path, result)
    print(f'{method} / {task}: {result["status"]} -> {output_path}')
    return result


def main():
    args = parse_args()
    defaults, entries = entries_from_args(args)
    if defaults:
        args.input_shape = [
            defaults.get('input_channels', args.input_shape[0]),
            defaults.get('input_height', args.input_shape[1]),
            defaults.get('input_width', args.input_shape[2]),
        ]
        args.batch_size = defaults.get('batch_size', args.batch_size)
        args.precision = defaults.get('precision', args.precision)
        args.warmup = defaults.get('warmup', args.warmup)
        args.iterations = defaults.get('iterations', args.iterations)
        args.repeat = defaults.get('repeat', args.repeat)
    os.makedirs(args.output_dir, exist_ok=True)
    results = [benchmark_one(entry, args, defaults) for entry in entries]
    save_json(os.path.join(args.output_dir, 'benchmark_run_results.json'),
              results)


if __name__ == '__main__':
    main()
