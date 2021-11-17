import os.path as osp
from collections import defaultdict

from mmedit.datasets.registry import DATASETS
from mmedit.datasets.base_dataset import BaseDataset


class BaseEnhanceDataset(BaseDataset):
    r"""General paired image folder dataset for image enhancement.

    It assumes that the all input images are in directory '/path/to/input'
    (`dir_lq`), and all groundtruth images are in directory '/path/to/gt'
    (`dir_gt`). A GT image has the SAME filename as the corresponding input
    image. The training and testing data are splitted using an annotation
    txt file (`ann_file`).
    Args:
        dir_lq (str): Path to the folder of input images.
        dir_gt (str): Path to the folder of groundtruth images.
        ann_file (str): Path to the annotation txt file.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool, optional): Store `True` when building test dataset.
            Default: `False`.
        filetmpl_lq (str, optional): Template for each filename for input images.
            Default: '{}.jpg'.
        filetmpl_gt (str, optional): Template for each filename for groundtruth images.
            Default: '{}.jpg'.
    """
    
    def __init__(self,
                 dir_lq,
                 dir_gt,
                 ann_file,
                 pipeline,
                 test_mode=False,
                 filetmpl_lq='{}.jpg',
                 filetmpl_gt='{}.jpg'):
        super().__init__(pipeline, test_mode=test_mode)
        
        if not osp.isfile(ann_file):
            raise ValueError('"ann_file" must be a path to annotation txt file.')
        
        self.dir_lq = dir_lq
        self.dir_gt = dir_gt
        self.ann_file = ann_file
        self.filetmpl_lq = filetmpl_lq
        self.filetmpl_gt = filetmpl_gt
        self.data_infos = self.load_annotations()
        
    def load_annotations(self):
        r"""Load annoations for enhancement dataset.

        It loads the LQ and GT image path from the annotation file.
        Each line in the annotation file contains the image name.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = self.filetmpl_lq.format(basename)
                gt_name = self.filetmpl_gt.format(basename)
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.dir_lq, lq_name),
                        gt_path=osp.join(self.dir_gt, gt_name)))
        return data_infos
    
    def evaluate(self, results, logger=None):
        r"""Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.
        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
    

@DATASETS.register_module()
class FiveK(BaseEnhanceDataset):
    pass


@DATASETS.register_module()
class PPR10K(BaseEnhanceDataset):
    r"""PPR10K dataset for image enhancement.

    The difference between this class and the base class is that multiple LQ
    images correspond to the same GT image.
    """
    
    def load_annotations(self):
        r"""Load annoations for enhancement dataset.

        It loads the LQ and GT image path from the annotation file.
        Each line in the annotation file contains the image name.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = self.filetmpl_lq.format(basename)
                gt_name = self.filetmpl_gt.format('_'.join(basename.split('_')[:2]))
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.dir_lq, lq_name),
                        gt_path=osp.join(self.dir_gt, gt_name)))
        return data_infos
    