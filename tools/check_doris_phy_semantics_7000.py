"""Check vegetation-group semantics in saved physiology checkpoint responses."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.doris_phy_evidence import RESPONSE_KEYS


SEMANTIC_PALETTE = [[238, 238, 238], [0, 158, 115], [213, 94, 0], [204, 121, 167], [0, 114, 178]]
VEGETATION_GROUPS = ('goji_correct', 'goji_as_vegetation', 'vegetation_as_goji', 'vegetation_correct')


def semantic_group_map(maps):
    """0 excludes comparison, 1 is GT vegetation, 2 is false vegetation."""
    reference = maps['reference_label']
    eligible = maps['valid_mask'].astype(bool) & np.isin(maps['pseudo_label'], [0, 1])
    groups = np.zeros(reference.shape, dtype=np.uint8)
    groups[eligible & np.isin(reference, [0, 1])] = 1
    groups[eligible & np.isin(reference, [2, 3, 4])] = 2
    return groups


def vegetation_confusion_map(maps):
    """Map GT-to-Teacher transitions within the two vegetation classes."""
    reference, prediction = maps['reference_label'], maps['pseudo_label']
    valid = maps['valid_mask'].astype(bool)
    groups = np.zeros(reference.shape,dtype=np.uint8)
    for code,(truth,predicted) in enumerate(((0,0),(0,1),(1,0),(1,1)),1):
        groups[valid & (reference==truth) & (prediction==predicted)] = code
    return groups


def response_group_statistics(maps, groups):
    rows = []
    for name, mask in groups:
        count = int(mask.sum())
        row = dict(group=name, valid_pixels=count)
        for key in RESPONSE_KEYS:
            values = maps[key][mask].astype(np.float64)
            row[key + '_mean'] = float(values.mean()) if count else None
            row[key + '_std'] = float(values.std()) if count else None
        row['downweighted_fraction'] = float((maps['delta_w_phy'][mask] < 0).mean()) if count else None
        rows.append(row)
    return rows


def semantic_statistics(maps):
    """Check both vegetation-class confusion and binary vegetation consistency."""
    metadata = json.loads(str(maps['metadata']))
    if metadata['capture_mode'] != 'checkpoint_replay':
        raise ValueError('Expected physiology checkpoint replay responses')
    evidence = metadata['phy_evidence']
    if evidence['class_names'] != ['goji', 'vegetation', 'building', 'water', 'road']:
        raise ValueError('Unexpected validation class mapping')
    reference = maps['reference_label']
    required = ('pseudo_label', 'valid_mask', 'm_phy') + RESPONSE_KEYS
    for key in required:
        if maps[key].shape != reference.shape:
            raise ValueError('Unaligned saved response: ' + key)
    if not np.isin(reference, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('Unexpected validation reference labels')
    valid = maps['valid_mask'].astype(bool)
    vegetation_prediction = np.isin(maps['pseudo_label'], [0, 1])
    if not np.array_equal(maps['m_phy'][valid].astype(bool), vegetation_prediction[valid]):
        raise ValueError('Saved physiology mask differs from Teacher vegetation predictions')
    for key in RESPONSE_KEYS:
        if not np.isfinite(maps[key][valid]).all():
            raise ValueError('Non-finite valid response: ' + key)
    np.testing.assert_allclose(maps['delta_w_phy'][valid],
        (maps['w_after_phy'] - maps['w_before_phy'])[valid], rtol=0, atol=1e-6)
    group_map = semantic_group_map(maps)
    eligible = group_map > 0
    groups = (
        ('gt_vegetation', group_map == 1),
        ('gt_nonvegetation', group_map == 2),
    )
    rows = response_group_statistics(maps,groups)
    comparison = dict(status='insufficient_groups',
        response_difference_false_minus_true=None, delta_difference_false_minus_true=None,
        false_vegetation_has_lower_response=None, false_vegetation_has_stronger_decrease=None)
    if all(row['valid_pixels'] for row in rows):
        true, false = rows
        response_difference = false['r_phy_mean'] - true['r_phy_mean']
        delta_difference = false['delta_w_phy_mean'] - true['delta_w_phy_mean']
        lower, stronger = response_difference < 0, delta_difference < 0
        comparison.update(
            status='consistent_direction' if lower and stronger else 'mixed_direction' if lower or stronger else 'expected_direction_not_observed',
            response_difference_false_minus_true=response_difference,
            delta_difference_false_minus_true=delta_difference,
            false_vegetation_has_lower_response=lower,
            false_vegetation_has_stronger_decrease=stronger)
    result = dict(data_source='checkpoint_replay', iter=metadata['iter'],
        checkpoint=metadata.get('checkpoint'), checkpoint_sha256=metadata.get('checkpoint_sha256'),
        sample_id=metadata['sample_id'], annotation_path=evidence.get('annotation_path'),
        annotation_sha256=evidence.get('annotation_sha256'),
        class_names=evidence['class_names'], ignore_index=255,
        group_definition='Valid Teacher predictions in {Goji, Vegetation}, grouped by GT vegetation membership; within-vegetation class errors remain in gt_vegetation',
        scope='One fixed validation image; descriptive pixel statistics with spatial dependence; no population significance or training benefit inferred',
        eligible_pixels=int(eligible.sum()),
        excluded_ignore_pixels=int((valid & vegetation_prediction & (reference == 255)).sum()),
        groups=rows, comparison=comparison)
    confusion = vegetation_confusion_map(maps)
    class_rows = response_group_statistics(maps,
        [(name,confusion==code) for code,name in enumerate(VEGETATION_GROUPS,1)])
    comparisons = []
    for name,correct,error in (('goji',class_rows[0],class_rows[1]),
                               ('vegetation',class_rows[3],class_rows[2])):
        pair = dict(gt_class=name,status='insufficient_groups',
            response_difference_error_minus_correct=None,delta_difference_error_minus_correct=None)
        if correct['valid_pixels'] and error['valid_pixels']:
            response_difference = error['r_phy_mean']-correct['r_phy_mean']
            delta_difference = error['delta_w_phy_mean']-correct['delta_w_phy_mean']
            pair.update(response_difference_error_minus_correct=response_difference,
                delta_difference_error_minus_correct=delta_difference,
                status='consistent_direction' if response_difference<0 and delta_difference<0
                    else 'expected_direction_not_observed')
        comparisons.append(pair)
    supported = sum(pair['status']=='consistent_direction' for pair in comparisons)
    status = 'both_directions_supported' if supported==2 else 'partially_supported' if supported else (
        'insufficient_groups' if any(pair['status']=='insufficient_groups' for pair in comparisons)
        else 'expected_direction_not_observed')
    result['primary_analysis'] = 'vegetation_class_check'
    result['vegetation_class_check'] = dict(groups=class_rows,comparisons=comparisons,status=status,
        group_definition='GT and Teacher both in {Goji, Vegetation}; compare correct and confused pixels within each GT class',
        excluded_pixels=int((valid & (confusion==0)).sum()),
        interpretation='Within-class association between physiology gating and Goji/other-vegetation confusion')
    return result


def export_semantic_check(source, output=None):
    source = Path(source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as maps:
        result = semantic_statistics(maps)
    result.update(source_npz=str(source), source_npz_sha256=digest,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stream = io.StringIO()
    check = result['vegetation_class_check']
    writer = csv.DictWriter(stream, fieldnames=list(check['groups'][0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(check['groups'])
    lines = [
        'Iter {} 生理门控地物语义核对'.format(result['iter']),
        '主要核对 Goji 和其他植被之间的区分，限定 GT 和 Teacher 均属于这两类的有效像素。',
        '每个真实类别内分别比较判对和被判为另一植被类别的像素。其他类别和 GT ignore=255 排除。',
        'ΔW=门控后权重−门控前权重，越负表示减权越强。',
    ]
    for row in check['groups']:
        lines.append('{} | n={} | R_phy={} | mean ΔW={}'.format(
            row['group'], row['valid_pixels'], row['r_phy_mean'], row['delta_w_phy_mean']))
    for pair in check['comparisons']:
        if pair['status']=='insufficient_groups':
            lines.append('GT {} 缺少判对或类间混淆像素，无法进行该方向比较。'.format(pair['gt_class']))
        else:
            lines.append('GT {} | 误判组减去判对组的平均 R_phy 差值为 {:.5f}，平均 ΔW 差值为 {:.5f} | {}'.format(
                pair['gt_class'],pair['response_difference_error_minus_correct'],pair['delta_difference_error_minus_correct'],
                '观察到误判组响应更低且减权更强' if pair['status']=='consistent_direction' else '未同时观察到误判组响应更低且减权更强'))
    lines += ['总体核对状态为 '+check['status'],
        '当前生理头把 Goji 和其他植被共同作为正类监督，本核对评估权重调整与两类混淆的关系。',
        '这是单张指定影像的描述性核对，像素存在空间相关性，结论适用范围限于该样本。',
        'ΔW由R_phy及门控规则计算，两者存在数学依赖。性能贡献结合独立消融指标判断。']
    contents = {'semantic_statistics.csv': stream.getvalue(),
        'semantic_statistics.json': json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        'semantic_summary.txt': '\n'.join(lines) + '\n'}
    output = Path(output).resolve() if output else source.parent/'phy_check'
    for name, content in contents.items():
        path = output/name
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError('Existing semantic check differs; retain it and use a separate response directory')
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Saved responses changed during analysis')
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = output/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.tmp')
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(path)
    return output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, help='saved responses.npz; defaults to the existing RGB-to-RGB Iter 7000 four-panel replay')
    args = parser.parse_args()
    source = args.responses
    if source is None:
        config = ROOT/'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py'
        assignments = ast.parse(config.read_text(encoding='utf-8')).body
        work_dir = next(ast.literal_eval(node.value) for node in assignments if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'work_dir' for target in node.targets))
        work = Path(work_dir)
        if not work.is_dir():
            work = ROOT/'shared-nvme'/work_dir.split('/shared-nvme/', 1)[1]
        folder = work/'analysis/fixed_probe/replayed/iter_7000_phy_paper_4panel/3_1024_1280_1280_1536'
        sources = list(folder.glob('iter_007000_*/responses.npz'))
        if len(sources) != 1:
            parser.error('Expected one saved replay; specify --responses explicitly')
        source = sources[0]
    output, result = export_semantic_check(source)
    print('PASS - saved-response semantic analysis completed')
    print('Result:', result['vegetation_class_check']['status'])
    for row in result['vegetation_class_check']['groups']:
        print(row['group'], 'n=', row['valid_pixels'], 'mean R_phy=', row['r_phy_mean'], 'mean delta=', row['delta_w_phy_mean'])
    print('Output:', output)


if __name__ == '__main__':
    main()
