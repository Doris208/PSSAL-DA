"""Check the flat eight-command launcher using an isolated training fixture."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('HPHY_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required for launcher execution tests')
class LauncherTests(unittest.TestCase):
    direction = 'goji26RGB2goji30RGB'
    launcher = 'scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(ROOT / 'outputs'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ('scripts', 'tools', 'bin'):
            (self.root / directory).mkdir(parents=True)
        source = (ROOT / self.launcher).read_text()
        self.source = source.replace(
            '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/' + self.direction, '$PWD/results')
        (self.root / 'scripts/run.sh').write_bytes(self.source.encode('utf-8'))
        direction = self.direction
        configs = self.root / 'configs/hrda/doris' / direction
        configs.mkdir(parents=True)
        for suffix in ('', '_phy_gate', '_struct_gate', '_phy_struct_gate',
                       '_ps_proto', '_struct_edge', '_pbcr', '_proto_struct_edge'):
            (configs / ('doris_' + direction + '_hrda_3vi' + suffix + '.py')).touch()
        command = self.root / 'bin/python'
        command.write_bytes(b'''#!/usr/bin/env bash
set -eu
[[ -f "$2" ]] || { echo "Missing configuration $2" >&2; exit 19; }
if [[ -d $4 && -n $(ls -A -- "$4") ]]; then
    echo "Experiment directory already exists: $4" >&2; exit 17
fi
mkdir -p -- "$4"
printf 'TRAIN %s\n' "$2"
printf '%s\n' "$2" > "$4/invoked"
printf 'Independent training log\n' > "$4/train.log"
if [[ ${HPHY_TEST_FAIL:-} == 1 && $2 == *_phy_gate.py ]]; then exit 23; fi
''')
        command.chmod(0o755)
        self.env = dict(os.environ)

    def run_script(self, *arguments):
        invocation = 'cd -- "$1"; export PATH="$PWD/bin:$PATH"; bash scripts/run.sh "${@:2}"'
        return subprocess.run([BASH, '-c', invocation, 'fixture', str(self.root), *arguments],
                              text=True, capture_output=True, env=self.env)

    def test_default_runs_all_eight_in_order_with_independent_training_logs(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        invoked = sorted((self.root / 'results').glob('*/invoked'))
        self.assertEqual(len(invoked), 8)
        self.assertEqual(len(list((self.root / 'results').glob('*/train.log'))), 8)
        self.assertEqual([line[6:] for line in result.stdout.splitlines() if line.startswith('TRAIN ')],
                         [shlex.split(line)[2] for line in self.source.splitlines()
                          if line.startswith('python tools/train.py ')])
        repeated = self.run_script()
        self.assertEqual(repeated.returncode, 17)
        self.assertIn('already exists', repeated.stderr)
        self.assertEqual(len(list((self.root / 'results').glob('*/invoked'))), 8)

    def test_one_copied_full_command_runs_independently_and_rejects_collision(self):
        command = next(line for line in self.source.splitlines()
                       if line.startswith('python tools/train.py ') and '_proto_struct_edge.py' in line)
        invocation = 'cd -- "$1"; export PATH="$PWD/bin:$PATH"; ' + command
        first = subprocess.run([BASH, '-c', invocation, 'fixture', str(self.root)],
                               text=True, capture_output=True, env=self.env)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(len(list((self.root / 'results').glob('*/invoked'))), 1)
        repeated = subprocess.run([BASH, '-c', invocation, 'fixture', str(self.root)],
                                  text=True, capture_output=True, env=self.env)
        self.assertEqual(repeated.returncode, 17)

    def test_training_failure_stops_immediately_at_second_command(self):
        self.env['HPHY_TEST_FAIL'] = '1'
        result = self.run_script()
        self.assertEqual(result.returncode, 23)
        self.assertIn('_phy_gate.py', result.stdout)
        self.assertEqual(len(list((self.root / 'results').glob('*/invoked'))), 2)

    def test_missing_config_and_retired_arguments_fail_before_training(self):
        for arguments in (('all', '--check'), ('2',)):
            self.assertEqual(self.run_script(*arguments).returncode, 2)
        self.assertFalse((self.root / 'results').exists())
        next(path for path in (self.root / 'configs').rglob('*.py')
             if path.name.endswith('_hrda_3vi.py')).unlink()
        result = self.run_script()
        self.assertEqual(result.returncode, 19)
        self.assertIn('Missing configuration', result.stderr)
        self.assertFalse((self.root / 'results').exists())

    def test_eight_explicit_commands_use_real_configs_and_distinct_workdirs(self):
        source = (ROOT / self.launcher).read_text()
        commands = [shlex.split(line) for line in source.splitlines()
                    if line.startswith('python tools/train.py ')]
        self.assertEqual(len(commands), 8)
        directories = []
        for command, suffix in zip(commands,
                ('', '_phy_gate', '_struct_gate', '_phy_struct_gate', '_ps_proto', '_struct_edge',
                 '_pbcr', '_proto_struct_edge')):
            config = 'configs/hrda/doris/' + self.direction + '/doris_' + self.direction + '_hrda_3vi' + suffix + '.py'
            self.assertEqual(command[2], config)
            self.assertTrue((ROOT / config).is_file())
            directory = command[command.index('--work-dir') + 1]
            self.assertEqual(directory, '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/'
                + self.direction + '/' + self.direction + '_hrda_3vi' + suffix)
            directories.append(directory)
        self.assertEqual(len(set(directories)), 8)
        self.assertNotIn('prepare_run', source)
        self.assertNotIn('for ', source)


class IRRGLauncherTests(LauncherTests):
    direction = 'goji26IRRG2goji30RGB'
    launcher = 'scripts/run_doris_tableiv_26irrg_to_30rgb_hphy15k.sh'


if __name__ == '__main__':
    unittest.main()
