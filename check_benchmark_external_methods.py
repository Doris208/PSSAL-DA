import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'benchmark_external_methods.json'
REQUIRED_FILES = [
    'tools/analysis/benchmark_external_methods.py',
    'tools/analysis/model_benchmark_adapter.py',
    'tools/analysis/model_efficiency_utils.py',
    'tools/analysis/aggregate_efficiency_results.py',
    'benchmark_external_methods.json',
    'run_benchmark_external_methods.sh',
]
TASKS = {
    'goji30IRRG2goji26IRRG',
    'goji26IRRG2goji30RGB',
    'goji26RGB2goji30RGB',
}
METHODS = {
    'Source Only',
    'DAFormer',
    'HRDA',
    'HRDA-R101',
    'MIC',
    'DACS',
    'SePiCo',
    'ProCA',
    'Ours / PSSAL-DA',
    'Ours / PSSAL-DA-MiT-B5',
}


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def load_manifest():
    with open(MANIFEST, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_files():
    for rel in REQUIRED_FILES:
        expect((ROOT / rel).exists(), f'Missing required file: {rel}')


def check_manifest():
    data = load_manifest()
    defaults = data['benchmark_defaults']
    expect(defaults['input_channels'] == 6, 'input_channels must be 6')
    expect(defaults['input_height'] == 256, 'input_height must be 256')
    expect(defaults['input_width'] == 256, 'input_width must be 256')
    expect(defaults['batch_size'] == 1, 'batch_size must be 1')
    expect(defaults['precision'].lower() == 'fp32', 'precision must be fp32')
    expect(defaults['warmup'] == 50, 'warmup must be 50')
    expect(defaults['iterations'] == 300, 'iterations must be 300')
    expect(defaults['repeat'] == 5, 'repeat must be 5')
    expect(defaults['flop_backend'] == 'local_hook_counter',
           'all methods must use local_hook_counter')

    seen = set()
    methods_by_name = {}
    for entry in data['methods']:
        key = (entry['name'], entry['task'])
        expect(key not in seen, f'Duplicate method/task entry: {key}')
        seen.add(key)
        expect(entry['task'] in TASKS, f'Unknown task: {entry["task"]}')
        expect(entry['name'] in METHODS, f'Unknown method: {entry["name"]}')
        expect((ROOT / entry['config']).exists(),
               f'Missing config: {entry["config"]}')
        expect(entry.get('adapter') == 'default',
               f'Unexpected adapter for {key}')
        checkpoint = entry.get('checkpoint')
        if checkpoint is not None:
            expect(Path(checkpoint).is_absolute() or (ROOT / checkpoint).exists(),
                   f'Checkpoint path is not usable: {checkpoint}')
        methods_by_name.setdefault(entry['name'], set()).add(entry['task'])

    for method in METHODS:
        expect(method in methods_by_name, f'Method not registered: {method}')
        expect(methods_by_name[method] == TASKS,
               f'Method {method} does not cover all three tasks')


def check_static_contents():
    adapter = (ROOT / 'tools/analysis/model_benchmark_adapter.py').read_text(
        encoding='utf-8')
    benchmark = (ROOT / 'tools/analysis/benchmark_external_methods.py').read_text(
        encoding='utf-8')
    utils = (ROOT / 'tools/analysis/model_efficiency_utils.py').read_text(
        encoding='utf-8')
    expect('build_segmentor(' in adapter,
           'benchmark adapter must build cfg.model deployment model')
    expect('build_train_model' not in adapter,
           'benchmark adapter must not build full UDA training wrapper')
    expect('encode_decode' in adapter,
           'forward adapter must prefer encode_decode')
    expect('local_hook_counter' in utils,
           'FLOP backend must be the unified hook counter')
    expect('torch_inference_context()' in utils and 'torch.no_grad()' in utils,
           'FPS measurement must use torch inference/no_grad context')
    expect('optimizer' not in benchmark,
           'benchmark must not run optimizer logic')


def main():
    check_files()
    check_manifest()
    check_static_contents()
    print('PASS: unified external-method efficiency benchmark is ready.')


if __name__ == '__main__':
    main()
