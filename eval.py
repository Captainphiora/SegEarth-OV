import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import segearth_segmentor
import custom_datasets
import custom_visualizer

from mmengine.config import Config
from mmengine.runner import Runner

from npu_compat import setup_npu
from utils import append_experiment_result


def parse_args():
    parser = argparse.ArgumentParser(
        description='SegEarth-OV evaluation with MMSeg')
    parser.add_argument('--config', default='./configs/cfg_voc20.py')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        default='',
        help='directory to save visualizaion images (empty = disable)')
    parser.add_argument(
        '--wait-time', type=float, default=2.0,
        help='the interval of show (s), only used with --show')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--num-levels', type=int, default=None,
                        help='override num_levels in feature_up_cfg (2/3/4)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

USE_PATCH = False
USE_INTERPOLATE_PATCH = False


def main():
    args = parse_args()
    setup_npu(use_patch=USE_PATCH, use_interpolate_patch=USE_INTERPOLATE_PATCH)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir

    # override num_levels if specified
    if args.num_levels is not None:
        cfg.model.feature_up_cfg.num_levels = args.num_levels

    # visualization
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    runner = Runner.from_cfg(cfg)
    results = runner.test()

    results.update({'VIT': cfg.model.vit_type,
                    'CLIP': cfg.model.clip_type,
                    'MODEL': cfg.model.model_type,
                    'Dataset': cfg.dataset_type})

    if runner.rank == 0:
        append_experiment_result('results.xlsx', [results])

    if runner.rank == 0:
        with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    main()
