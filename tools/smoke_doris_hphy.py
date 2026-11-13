"""Short full HRDA/DACS synthetic acceptance, never a formal training run."""
import argparse
import gc
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmcv.runner import build_optimizer, save_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_doris_hphy_configs import EXPERIMENTS, check_config, config_path
from tools.doris_probe_data import RGB30_DIRECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--direction', choices=RGB30_DIRECTIONS, default=RGB30_DIRECTIONS[0])
    parser.add_argument('--experiments', nargs='+', type=int, choices=range(1, 9), default=[2, 4, 8])
    parser.add_argument('--output', required=True, help='New directory, refused if it already exists')
    parser.add_argument('--check-fixed-probes', action='store_true',
                        help='Initialize production collectors and export real fixed-input probes using the configured dataset')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. GPU acceptance must run on a CUDA host.')
    output = Path(args.output).resolve()
    if 'doris_stage1' in output.parts:
        raise ValueError('Smoke output must be isolated from historical experiments')
    output.mkdir(parents=True, exist_ok=False)
    from mmseg.models.builder import build_train_model
    from mmseg.utils.mechanism import branch_probe_payload, model_digest, readonly_probe
    from mmseg.models.uda.phy_reliability import phy_gate_strength

    torch.set_num_threads(2)
    report = dict(device=str(device), direction=args.direction, torch=torch.__version__, synthetic=True,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  initialization='Random model weights, pretrained paths disabled only in this smoke',
                  teacher_fixture='Real EMA network with controlled classifier biases to exercise high-confidence target supervision',
                  experiments=[])
    for number in args.experiments:
        name, suffix, flags = EXPERIMENTS[number - 1]
        print('{} constructing full HRDA/DACS model'.format(name), flush=True)
        cfg, audit = check_config(config_path(suffix, args.direction), flags)
        cfg.model.pretrained = None
        cfg.model.backbone.pretrained = None
        cfg.model.train_cfg.work_dir = str(output)
        cfg.uda.update(mechanism_logging=True, mechanism_sample_interval=1,
                       mechanism_output_mode='branches', mechanism_detailed_output=True)
        torch.manual_seed(0)
        np.random.seed(0)
        model = build_train_model(cfg).to(device)
        model.init_weights()
        model._init_ema_weights()
        optimizer = build_optimizer(model, cfg.optimizer)
        image = torch.randn(2, 6, 256, 256, device=device)
        vi = torch.rand(2, 3, 256, 256, device=device)
        gt = ((torch.arange(256, device=device) // 32) % 5)[None, None, None, :].expand(2, 1, 256, 256).clone()
        gt[:, :, :3] = 255
        norm = dict(mean=[0.] * 6, std=[1.] * 6, to_rgb=False)
        metas = [dict(filename='synthetic_{}.tif'.format(i), ori_shape=(256, 256, 6),
                      img_shape=(256, 256, 6), pad_shape=(256, 256, 6),
                      scale_factor=1., flip=False, img_norm_cfg=norm) for i in range(2)]
        valid = image.new_zeros(2, 1, 256, 256)
        valid[:, :, 30:226, 30:226] = 1
        data = dict(img=image, target_img=torch.randn_like(image), img_metas=metas,
                    target_img_metas=metas, gt_semantic_seg=gt,
                    source_vi_img=vi, target_vi_img=torch.rand_like(vi), valid_pseudo_mask=valid)
        runs = []
        # Observe the real DACS/ClassMix loss path without changing its tensors.
        trace = {}
        scope = model.forward_train.__func__.__globals__
        original_masks, original_record = scope['get_class_masks'], scope['training_record']
        original_forward = model.get_model().forward_train

        def capture_masks(*inputs, **kwargs):
            masks = original_masks(*inputs, **kwargs)
            trace['mix_mask'] = torch.cat(masks)[:, 0].detach().clone().bool()
            return masks

        def capture_record(*inputs, **kwargs):
            trace['final_weight'] = inputs[4].detach().clone()
            response = kwargs.get('r_phy')
            if isinstance(response, dict):
                trace['phy'] = {key: value.detach().clone() for key, value in response.items()}
            return original_record(*inputs, **kwargs)

        def capture_forward(*inputs, **kwargs):
            if len(inputs) > 3 and inputs[3] is not None:
                trace['mixed_weight'] = inputs[3].detach().clone()
            return original_forward(*inputs, **kwargs)

        scope['get_class_masks'], scope['training_record'] = capture_masks, capture_record
        model.get_model().forward_train = capture_forward
        # Early nonzero, the configured full-strength point, then all other branches active.
        for iteration, teacher_class in ((0, 0), (4000, 1), (10000, 2)):
            print('{} forward/backward at {}'.format(name, iteration), flush=True)
            trace.clear()
            model.local_iter = iteration
            # Random teachers usually select no pixel at tau=.968. Controlled
            # classifier weights exercise both target BCE groups and the active
            # gate while retaining the real Teacher forward and EMA update.
            with torch.no_grad():
                classifier = model.get_ema_model().decode_head.head.conv_seg
                classifier.weight.zero_()
                classifier.bias.fill_(-10)
                classifier.bias[teacher_class] = 10
                if iteration == 0:
                    # DACS initializes EMA from Student on its first real step.
                    student_classifier = model.get_model().decode_head.head.conv_seg
                    student_classifier.weight.copy_(classifier.weight)
                    student_classifier.bias.copy_(classifier.bias)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                result = model.train_step(data, optimizer)
            logs = result['log_vars']
            if not all(np.isfinite(value) for value in logs.values()):
                raise AssertionError('Non-finite loss or diagnostic in ' + name)
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise AssertionError('Non-finite parameter in ' + name)
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise AssertionError('Non-finite gradient in ' + name)
            if flags[0] and not any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.phy_head.parameters()):
                raise AssertionError('H-Phy source gradient missing in ' + name)
            if flags[0] and logs['loss_phy_tgt_veg_weighted'] <= 0:
                raise AssertionError('Controlled Teacher failed to exercise target vegetation BCE')
            if flags[1]:
                gate = model.struct_gate
                if not gate.source_adj_initialized.any() or not gate.semantic_proto_initialized.any():
                    raise AssertionError('Source structural EMA priors were not initialized')
                expected_strength = .10 + .90 * min(iteration / 4000., 1.)
                if (abs(gate.gate_strength(iteration) - expected_strength) > 1e-7 or
                        gate.feature_strength(iteration, 6000, 4000) != gate.gate_strength(iteration)):
                    raise AssertionError('Structural supervision and feature schedules differ')
            expected_mixed = torch.where(trace['mix_mask'], torch.ones_like(trace['final_weight']),
                                         trace['final_weight'])
            torch.testing.assert_close(trace['mixed_weight'], expected_mixed)
            if flags[0]:
                rho = phy_gate_strength(iteration, **model.phy_gate_params)
                if abs(logs['phy_gate_rho'] - rho) > 1e-7:
                    raise AssertionError('Logged strength differs from the actual local_iter schedule')
                phy = trace['phy']
                torch.testing.assert_close(phy['w_after_phy'], phy['w_before_phy'] * phy['g_phy'])
                torch.testing.assert_close(phy['a_phy'], 1-phy['g_phy'])
                if iteration <= 4000:
                    torch.testing.assert_close(trace['final_weight'], phy['w_after_phy'][:, 0])
                    if not logs['phy_gate_active_ratio'] > 0:
                        raise AssertionError('Early gate failed to change actual target supervision')
                for head in (model.phy_head.veg_head, model.phy_head.gv_head):
                    if head.weight.grad is None or head.weight.grad.abs().sum() == 0:
                        raise AssertionError('A source H-Phy head has no gradient')
            runs.append(dict(iteration=iteration, logs=logs, classmix_weight_identity=True,
                             phy_weight_identity=flags[0]))
        scope['get_class_masks'], scope['training_record'] = original_masks, original_record
        model.get_model().forward_train = original_forward
        # Same-model checkpoint/optimizer restoration, including all branch banks.
        with tempfile.TemporaryDirectory(prefix='checkpoint_', dir=str(output)) as temporary:
            from mmseg.utils.checkpoint_policy import prepare_save
            checkpoint_path = Path(temporary) / ('iter_{}.pth'.format(model.local_iter))
            runner = SimpleNamespace(model=model, optimizer=optimizer, meta={}, iter=model.local_iter-1)
            prepare_save(runner, checkpoint_path)
            save_checkpoint(model, str(checkpoint_path), optimizer=optimizer,
                            meta=dict(runner.meta, config=cfg.pretty_text, iter=model.local_iter,
                                      head_version=getattr(model, 'phy_mode', 'disabled'),
                                      phy_cfg=dict(model.phy_cfg), phy_gate=getattr(model, 'phy_gate_params', {})))
            before = model_digest(model)
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if model_digest(model) != before:
                raise AssertionError('Checkpoint roundtrip changed model state')
            del checkpoint
        metadata = dict(iter=model.local_iter, sample_id=metas[0]['filename'],
                        config_sha256=name, synthetic_fixture=True, capture_mode='synthetic_smoke',
                        modules=dict(zip(('phy', 'struct', 'proto', 'edge'), flags)),
                        states={}, input_view=['R', 'G', 'B'])
        payload = branch_probe_payload(model, image[:1], vi[:1], metas[:1], valid[:1], metadata)
        np.savez_compressed(str(output / (name.lower() + '_probe.npz')), **payload)
        if flags[1]:
            required = {'r_conf', 'r_loc', 'r_adj', 'r_prob', 'r_struct',
                        'adjacency_available', 'semantic_available', 'evidence_count',
                        'w_before_struct', 'w_after_struct', 'w_final'}
            if required - set(payload):
                raise AssertionError('Structural V2 probe fields missing')
        if args.check_fixed_probes and any(flags):
            from mmseg.utils.mechanism import MechanismCollector
            runner = SimpleNamespace(iter=model.local_iter-1, work_dir=str(output / name.lower()),
                meta=dict(config=cfg.pretty_text), mechanism_config=cfg, resume_from=None,
                logger=logging.getLogger('smoke'))
            collector = MechanismCollector(runner, model)
            collector.snapshot(model.local_iter)
            del collector, runner
        with readonly_probe(model):
            prediction = model.get_model().simple_test(image[:1], metas[:1], rescale=True)
        if len(prediction) != 1 or prediction[0].shape != (256, 256):
            raise AssertionError('Student validation output has the wrong shape')
        report['experiments'].append(dict(audit, name=name, runs=runs,
            parameters=sum(p.numel() for p in model.parameters()),
            hphy_parameters=sum(p.numel() for p in model.phy_head.parameters()) if flags[0] else 0,
            checkpoint_roundtrip=True, checkpoint_space_check=True,
            student_validation=True, probe_export=True, fixed_probe_collector=args.check_fixed_probes and any(flags)))
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('{} PASS on {}'.format(name, device), flush=True)
        del original_forward, capture_forward, capture_masks, capture_record, scope
        del model, optimizer, data, image, vi, gt, valid, payload
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
