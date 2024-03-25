

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from os import listdir
from glob import glob
import argparse

from eval_toolkit.datasets import DatasetFactory
from eval_toolkit.evaluation import OPEBenchmark, EAOBenchmark
from eval_toolkit.visualization.draw_success_precision import draw
from configs.DataPath import get_root, SYSTEM

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--dataset', '-d', default='DTB70', type=str, help='dataset name')
args = parser.parse_args()


def main():
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=get_root(args.dataset), load_img=False)
    base_name = dataset.base_name
    tracker_dir = os.path.join('results', base_name)

    trackers = []
    trackers = ['MobileSiam-LT', 'SiamGAT', 'SiamRPNpp']

    dataset.set_tracker(tracker_dir, trackers)

    if 'VOT20' in args.dataset and 'VOT2020' not in args.dataset:
        benchmark = EAOBenchmark(dataset, tags=dataset.tags)
    else:
        benchmark = OPEBenchmark(dataset)

    videos = list(dataset.videos.keys())
    videos.sort()
    for test_video in videos:
        draw(dataset=benchmark.dataset, dataset_name=dataset.base_name, video_name=test_video,
             eval_trackers=trackers, draw_gt=False, save=True, wait_key=5, width=12, font=7)


if __name__ == '__main__':
    vis_tune()
    main()
