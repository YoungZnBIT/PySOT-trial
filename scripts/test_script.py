

import numpy as np
import torch

from configs.DataPath import get_root, SYSTEM
from configs.get_config import get_config, Config
from eval_toolkit.datasets import DatasetFactory
from eval_toolkit.evaluation import OPEBenchmark, EAOBenchmark
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.models.model.model_builder import build_model
from pysot.trackers.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from pysot.models.backbone.repvgg import repvgg_model_convert
from eval_toolkit.utils.test import test

import argparse
import os
import logging

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamese tracking')


parser.add_argument('--dataset', default='DTB70', type=str, help='name of dataset')
# parser.add_argument('--tracker', default='MobileSiam', type=str, help='config file')
# parser.add_argument('--config', default='experiments/mobilesiam/mobilesiam-st.yaml', type=str, help='config file')
# parser.add_argument('--snapshot', default='experiments/mobilesiam/MobileSiam-ST.pth', type=str, help='model name')
parser.add_argument('--tracker', default='UPDMobileSiam', type=str, help='config file')
parser.add_argument('--config', default='experiments/mobilesiam/mobilesiam-lt.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='experiments/mobilesiam/MobileSiam-LT.pth', type=str, help='model name')
parser.add_argument('--gpu_id', default=1, type=int, help="gpu id")
parser.add_argument('--result_path', default='results', type=str, help='results path')  # 非tune模式时结果保存的文件夹
parser.add_argument('--save', default='base', type=str, help='save manner')  # 只在数据集中的一部分序列上进行测试时，选择保存结果文件的方式
parser.add_argument('--trk_cfg', default='', type=str, help='track config')  # 输入此次测试时想使用的跟踪参数
parser.add_argument('--test_name', default='', type=str, help='test name')  # 本次测试名，用于消融实验、参数实验等，如para-0.30-0.10

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

"""          
python scripts/test_script.py --dataset UAV10fps --tracker MobileSiam --config experiments/mobilesiam/trial-a2.yaml --snapshot snapshot/Mobile-A2-test2/checkpoint_e40.pth --save base --test_name Mobile-A2-test2 --gpu_id 1
"""


def test_all(tracker, name, track_cfg, dataset, save_path='results', visual=False, test_name=''):
    cfg.TRACK.CONTEXT_AMOUNT = track_cfg.context_amount
    cfg.TRACK.WINDOW_INFLUENCE = track_cfg.window_influence
    cfg.TRACK.PENALTY_K = track_cfg.penalty_k
    cfg.TRACK.LR = track_cfg.size_lr
    if 'CONFIDENCE' in cfg.TRACK:
        cfg.TRACK.CONFIDENCE = track_cfg.confidence
    if 'UPDATE_FREQ' in cfg.TRACK:
        cfg.TRACK.UPDATE_FREQ = track_cfg.update_freq
    test(tracker, name, dataset, test_video='', save_path=save_path, visual=visual, test_name=test_name)
    results = evaluate(dataset, name, save_path, test_name=test_name)
    print('{:s} results: {:.4f}'.format(name, results))


def evaluate(dataset, tracker_name, result_path='results', test_name=''):
    if test_name == '':
        tracker_dir = os.path.join(result_path, save_name)
    else:
        tracker_dir = os.path.join(result_path, test_name + '-' + save_name)
    trackers = [tracker_name]
    dataset.set_tracker(tracker_dir, trackers)

    if 'VOT20' in args.dataset and 'VOT2020' not in args.dataset:
        benchmark = EAOBenchmark(dataset, tags=dataset.tags)
        results = benchmark.eval(trackers)
        eao = results[tracker_name]['all']
        return eao
    elif 'ITB' in args.dataset:
        benchmark = OPEBenchmark(dataset)
        mIou_ret, mIou_scen = benchmark.eval_mIoU()
        mIoU = np.mean(list(mIou_ret[tracker_name].values()))
        return mIoU
    else:
        benchmark = OPEBenchmark(dataset)
        cle = benchmark.eval_cle(trackers)
        success_ret = benchmark.eval_success(trackers)
        auc = np.mean(list(success_ret[tracker_name].values()))
        return auc


def obj(trial):
    track_cfg.context_amount = trial.suggest_float('context_amount', 0.45, 0.55)
    # track_cfg.context_amount = trial.suggest_float('context_amount', 0.45, 0.51)
    # track_cfg.context_amount = 0.5
    track_cfg.window_influence = trial.suggest_float('window_influence', 0.25, 0.60)
    # track_cfg.window_influence = trial.suggest_float('window_influence', 0.40, 0.60)
    # track_cfg.window_influence = 0.35
    track_cfg.penalty_k = trial.suggest_float('penalty_k', 0.02, 0.18)
    # track_cfg.penalty_k = trial.suggest_float('penalty_k', 0.08, 0.17)
    # track_cfg.penalty_k = 0.06
    track_cfg.size_lr = trial.suggest_float('scale_lr', 0.25, 0.60)
    # track_cfg.size_lr = trial.suggest_float('scale_lr', 0.25, 0.40)
    # track_cfg.size_lr = 0.30

    if 'CONFIDENCE' in cfg.TRACK:
        # track_cfg.confidence = trial.suggest_float('confidence', 0.05, 0.95)
        track_cfg.confidence = trial.suggest_float('confidence', 0.1, 0.85)
        # track_cfg.confidence = 0.
    else:
        track_cfg.confidence = 0.

    if 'UPDATE_FREQ' in cfg.TRACK:
        track_cfg.update_freq = trial.suggest_int('update_freq', 5, 20)
        # track_cfg.update_freq = 0
    else:
        track_cfg.update_freq = 0

    name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}_upd-{:d}'.format(
        model_name, track_cfg.context_amount, track_cfg.window_influence,
        track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence, track_cfg.update_freq)

    test_all(tracker, name, track_cfg, dataset, save_path=tune_root, test_name=test_name)
    results = evaluate(dataset, name, result_path=tune_root, test_name=test_name)
    logger.info("{:s} Results: {:.3f}, context_amount: {:.7f}, window_influence: {:.7f}, penalty_k: {:.7f}, "
                "lr: {:.7f}, confidence: {:.7f}, update_freq: {:d}".
                format(model_name, results, track_cfg.context_amount, track_cfg.window_influence,
                       track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence, track_cfg.update_freq))
    return results


def tune():
    import optuna

    db_root = tune_root + 'A-tune-dbs/'
    if not os.path.exists(db_root):
        os.makedirs(db_root)
    if SYSTEM == 'Windows':
        root_path = os.getcwd()
        db_root = os.path.join(root_path, db_root)

    log_root = tune_root + 'logs/'
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    init_log('global', logging.INFO)

    if test_name == '':
        log_path = log_root + '{:s}-tune-logs.txt'.format(model_name)
        db_path = db_root + '{:s}-tune.db'.format(model_name)
        db_path_ = "sqlite:///" + db_path

        add_file_handler('global', log_path, logging.INFO)
        if not os.path.exists(db_path):
            study = optuna.create_study(study_name="{:s}".format(model_name), direction='maximize', storage=db_path_)
        else:
            study = optuna.load_study(study_name="{:s}".format(model_name), storage=db_path_)
    else:
        log_path = log_root + '{:s}-{:s}-tune-logs.txt'.format(test_name, model_name)
        db_path = db_root + '{:s}-{:s}-tune.db'.format(test_name, model_name)
        db_path_ = "sqlite:///" + db_path

        add_file_handler('global', log_path, logging.INFO)
        if not os.path.exists(db_path):
            study = optuna.create_study(study_name="{:s}-{:s}".format(test_name, model_name), direction='maximize', storage=db_path_)
        else:
            study = optuna.load_study(study_name="{:s}-{:s}".format(test_name, model_name), storage=db_path_)
    study.optimize(obj, n_trials=10000)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))


def set_cfg(trk_cfg, add_cfg):
    add_cfg = add_cfg.split(', ')
    for i in range(len(add_cfg)):
        new_cfg = add_cfg[i].split(': ')
        if new_cfg[0] == 'lr':
            new_cfg[0] = 'size_lr'
        setattr(trk_cfg, new_cfg[0], float(new_cfg[1]))
    return trk_cfg


if __name__ == '__main__':
    # init
    test_name = args.test_name
    tracker_name = args.tracker

    cfg = get_config(tracker_name)
    # load and merge config, 主要是模型设置相关参数
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    # device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = build_model(tracker_name, cfg)

    # UPDMobileSiam的训练经过了多个阶段, 最后一个阶段要冻结所有部分, 单独训练confidence head, 比较特殊, 因此加载权重较麻烦
    if tracker_name == 'UPDMobileSiam' and hasattr(model, 'process_conf'):
        model.backbone = repvgg_model_convert(model.backbone)
        model.head = repvgg_model_convert(model.head)
        model.process_zf = repvgg_model_convert(model.process_zf)
        model.updater = repvgg_model_convert(model.updater)
        if torch.cuda.is_available():
            model = load_pretrain(model, args.snapshot).cuda().eval()
        else:
            model = load_pretrain(model, args.snapshot, False).eval()
        model.process_conf = repvgg_model_convert(model.process_conf)

    # 其他正常模型的加载过程
    else:
        # first load backbone if it's a repvgg network
        if 'RepVGG' in cfg.BACKBONE.TYPE and cfg.BACKBONE.PRETRAINED and cfg.BACKBONE.TRAIN_EPOCH >= cfg.TRAIN.EPOCH:
            # load_pretrain(model.backbone, cfg.BACKBONE.PRETRAINED)
            repvgg_model_convert(model.backbone, do_copy=False)

        # load model
        if torch.cuda.is_available():
            model = load_pretrain(model, args.snapshot).cuda().eval()
        else:
            model = load_pretrain(model, args.snapshot, False).eval()

        model = repvgg_model_convert(model)

    # 统计Params and GFlops, 需要model实现forward_param()方法
    # from thop import profile 
    # model = model.cpu()
    # if args.tracker == 'TransT':
    #     model.forward_param_z()
    # x = torch.rand((1, 3, 255, 255))  # (1, 3, 256, 256)
    # flops, params = profile(model, inputs=(x,))
    # gflpops = flops / 1e9
    # params_ = params / 1e6
    # model = model.cuda()

    # build trackers
    tracker = build_tracker(cfg, model)

    # dataset_ = 'VOT2018'
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=get_root(args.dataset), load_img=False)
    dataset.save = args.save
    dataset_name = dataset.name
    base_name = dataset.base_name
    if dataset.save == 'base' or dataset.save == 'all':
        save_name = base_name
    elif dataset.save == 'derive':
        save_name = dataset_name
    model_name = tracker_name + '-' + args.snapshot.split('/')[-1].split('.')[0] + '-' + save_name
    print('test model name: {:s}'.format(model_name))

    # 评估测试时最重要的  跟踪器相关参数  的设置
    track_cfg = Config()
    track_cfg.context_amount = cfg.TRACK.CONTEXT_AMOUNT
    track_cfg.window_influence = cfg.TRACK.WINDOW_INFLUENCE
    track_cfg.penalty_k = cfg.TRACK.PENALTY_K
    track_cfg.size_lr = cfg.TRACK.LR
    if 'CONFIDENCE' in cfg.TRACK:
        track_cfg.confidence = cfg.TRACK.CONFIDENCE
    else:
        track_cfg.confidence = 0.
    if 'UPDATE_FREQ' in cfg.TRACK:
        track_cfg.update_freq = cfg.TRACK.UPDATE_FREQ
    else:
        track_cfg.update_freq = 0
    if args.trk_cfg != '':
        track_cfg = set_cfg(track_cfg, args.trk_cfg)

    """
    Test Mode 0: Tune
    """
    tune_root = 'tune_results/'
    # tune()

    """
    Test Mode 1: Evaluate the performance of a tracker with corresponding config on the chosen dataset
    """
    # # Mode 1.1: 调参时使用, 便于测试给定参数组合下的跟踪器
    # name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}_upd-{:d}'.format(
    #     model_name, track_cfg.context_amount, track_cfg.window_influence,
    #     track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence, track_cfg.update_freq)
    # test_all(tracker, name=name, track_cfg=track_cfg, dataset=dataset, save_path=args.result_path, test_name=test_name)
    # # Mode 1.2: 可视化观察, debug常用
    test_all(tracker, name=tracker_name, track_cfg=track_cfg, dataset=dataset, visual=True, save_path='')
    # Mode 1.3: 非可视化观察, 可统计跟踪速度
    # test_all(tracker, name=tracker_name, track_cfg=track_cfg, dataset=dataset, visual=False, save_path='')
    # # Mode 1.4: 评估测试模式, 遍历并保存
    # test_all(tracker, name=model_name, track_cfg=track_cfg, dataset=dataset, visual=False, save_path=args.result_path)
