"""Exercise the flat offline commands with isolated Python command fixtures."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('HPHY_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required for offline launcher execution tests')
class OfflineShellTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for directory in ('scripts', 'bin'):
            (self.root / directory).mkdir(parents=True)
        self.original = (ROOT / 'scripts/run_hphy_mechanism_26rgb2_30rgb.sh').read_text()
        self.source = self.original
        (self.root / 'scripts/run.sh').write_bytes(self.source.encode())
        (self.root / 'bin/python').write_bytes(b'''#!/usr/bin/env bash
set -eu
printf '%s\n' "$1" >> "$PWD/commands.log"
if [[ ${SYNTHETIC_FAIL:-} == replay && $1 == tools/replay_hphy_mechanism.py ]]; then exit 31; fi
mkdir -p "$PWD/results"
out=$(mktemp -d "$PWD/results/run_XXXXXX")
mkdir -p "$out/logs"
if [[ $1 == tools/replay_hphy_mechanism.py ]]; then
    touch "$out/responses_full_256.npz" "$out/phy_mechanism_meta.json" "$out/logs/replay.log"
    if [[ ${SYNTHETIC_FAIL:-} == render ]]; then exit 32; fi
    touch "$out/render_complete" "$out/logs/render.log"
fi
''')
        (self.root / 'bin/python').chmod(0o755)
        self.env = dict(os.environ)
        self.output = self.root / 'results'

    def run_script(self, *arguments):
        invocation = 'cd -- "$1"; export PATH="$PWD/bin:$PATH"; bash scripts/run.sh "${@:2}"'
        return subprocess.run([BASH, '-c', invocation, 'fixture', str(self.root), *arguments],
                              text=True, capture_output=True, env=self.env)

    def test_single_short_command_uses_python_defaults_and_exports_figures(self):
        commands = [shlex.split(line) for line in self.original.splitlines() if line.startswith('python ')]
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0], ['python', 'tools/replay_hphy_mechanism.py'])
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.output.glob('*/render_complete'))), 1)
        self.assertEqual(len(list(self.output.glob('*/logs/replay.log'))), 1)
        self.assertEqual(len(list(self.output.glob('*/logs/render.log'))), 1)
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(len(list(self.output.glob('*/render_complete'))), 2)

    def test_copied_inference_command_also_exports_figures(self):
        command = next(line for line in self.source.splitlines() if line.startswith('python '))
        invocation = 'cd -- "$1"; export PATH="$PWD/bin:$PATH"; ' + command
        result = subprocess.run([BASH, '-c', invocation, 'fixture', str(self.root)],
                                text=True, capture_output=True, env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'commands.log').read_text().splitlines(),
                         ['tools/replay_hphy_mechanism.py'])
        self.assertEqual(len(list(self.output.glob('*/render_complete'))), 1)

    def test_replay_failure_and_retired_arguments_stop_execution(self):
        self.assertEqual(self.run_script('run').returncode, 2)
        self.assertFalse((self.root / 'commands.log').exists())
        self.env['SYNTHETIC_FAIL'] = 'replay'
        result = self.run_script()
        self.assertEqual(result.returncode, 31)
        self.assertEqual((self.root / 'commands.log').read_text().splitlines(),
                         ['tools/replay_hphy_mechanism.py'])

    def test_renderer_failure_keeps_exit_status(self):
        self.env['SYNTHETIC_FAIL'] = 'render'
        self.assertEqual(self.run_script().returncode, 32)
        self.assertFalse(list(self.output.glob('*/render_complete')))


if __name__ == '__main__':
    unittest.main()
