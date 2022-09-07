import os.path as osp
import mmcv
from mmcv import Config
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets import build_dataset
from mmseg.apis import set_random_seed
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

classes = (
    'sky', 'Bulding', 'Pole', 'Road_marking', 'Road', 'Pavement', 'Tree',
    'SingSymbole', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
)

palette = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [255, 69, 0], [128, 64, 128], [60, 40, 222],
    [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]
]


@DATASETS.register_module()
class SegnetTutorialDataset(CustomDataset):

    CLASSES = classes
    PALETTE = palette

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)


def main():

    cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_480x360_segnet_tutorial.py')

    cfg.work_dir = './work_dirs/segnet_tutorial'
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'
    datasets = [build_dataset(cfg.data.train)]
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')
    )

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())


if __name__ == '__main__':
    main()
