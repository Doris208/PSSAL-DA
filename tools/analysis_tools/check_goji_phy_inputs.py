import argparse

from mmcv import Config

from mmseg.datasets import build_dataset


def _shape(data):
    if hasattr(data, 'data'):
        data = data.data
    return tuple(data.shape)


def main():
    parser = argparse.ArgumentParser(
        description='Check Goji physiology-gate dataset inputs.')
    parser.add_argument('config', help='Path to a phy gate config file.')
    parser.add_argument('--split', default='source', choices=['source',
                                                             'target'])
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train[args.split])
    sample = dataset[0]
    print(f'{args.split} img shape: {_shape(sample["img"])}')
    print(f'{args.split} vi_img shape: {_shape(sample["vi_img"])}')
    if 'gt_semantic_seg' in sample:
        print(f'{args.split} gt shape: {_shape(sample["gt_semantic_seg"])}')


if __name__ == '__main__':
    main()
