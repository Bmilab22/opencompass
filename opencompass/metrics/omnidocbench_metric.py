from collections import defaultdict
from typing import Optional, Dict, List, Any
from tabulate import tabulate
import pandas as pd

from mmengine.evaluator import BaseMetric
from opencompass.registry import METRICS


@METRICS.register_module()
class OmniDocBenchMetric(BaseMetric):
    """Custom metric for evaluating model predictions with additional result processing.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
    """

    # task_dict = {
    #     'Perception': [ 'OCR'],
    # }

    # def __init__(self,
    #              collect_device: str = 'cpu',
    #              prefix: Optional[str] = None) -> None:
    #     super().__init__(collect_device, prefix)

    # def process(self, data_batch, data_samples) -> None:
    #     for data_sample in data_samples:
    #         result = dict()
    #         result['img_path'] = data_sample['img_path']
    #         result['task'] = data_sample['task']
    #         result['pred'] = 1 if data_sample['answer'].lower(
    #         ) == data_sample['pred_answer'].lower() else 0
    #         result['metric'] = {}  # Initialize an empty metric dictionary for each sample
    #         self.results.append(result)

    # def compute_metrics(self, results: list, page_info: Dict[str, Any]) -> Dict[str, Any]:
    #     # Original computation logic
    #     record = {task: defaultdict(int) for task in self.task_dict['Perception'] + self.task_dict['Cognition']}
    #     for sample in results:
    #         record[sample['task']][sample['img_path']] += sample['pred']

    #     metric = {}
    #     for task in self.task_dict['Perception'] + self.task_dict['Cognition']:
    #         single_sum, double_sum = 0., 0.
    #         for v in record[task].values():
    #             assert 0 <= v <= 2
    #             if v == 2:
    #                 single_sum += 2
    #                 double_sum += 1
    #             elif v == 1:
    #                 single_sum += 1
    #         acc = single_sum / 2 / len(record[task])
    #         acc_plus = double_sum / len(record[task])

    #         metric[task] = {
    #             'acc': acc,
    #             'acc_plus': acc_plus,
    #             'score': 100 * (acc + acc_plus)
    #         }

    #     # Compute overall score
    #     metric['Perception'] = sum(metric[task]['score'] for task in self.task_dict['Perception'])
    #     metric['Cognition'] = sum(metric[task]['score'] for task in self.task_dict['Cognition'])
    #     metric['Overall'] = metric['Perception'] + metric['Cognition']

    #     # Additional processing functions
    #     full_labels_results = self.get_full_labels_results(results)
    #     page_split_results = self.get_page_split(results, page_info)

    #     # Combine all results
    #     combined_results = {'OriginalMetrics': metric, **full_labels_results, **page_split_results}

    #     # Display results
    #     self.show_result(combined_results)

    #     return combined_results

    # @staticmethod
    # def show_result(results: Dict[str, Any]):
    #     for metric_name in results.keys():
    #         print(f'{metric_name}:')
    #         score_table = [[k, v] for k, v in results[metric_name].items()]
    #         print(tabulate(score_table))
    #         print('='*100)

    # @staticmethod
    # def sort_nested_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
    #     if isinstance(d, dict):
    #         sorted_dict = {k: MMEMetric.sort_nested_dict(v) for k, v in sorted(d.items())}
    #         return sorted_dict
    #     return d

    # @staticmethod
    # def get_full_labels_results(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     label_group_dict = defaultdict(lambda: defaultdict(list))
    #     for sample in samples:
    #         label_list = []
    #         if not sample.get("gt_attribute"):
    #             continue
    #         for anno in sample["gt_attribute"]:
    #             for k, v in anno.items():
    #                 label_list.append(f"{k}: {v}")
    #         for label_name in set(label_list):
    #             for metric, score in sample['metric'].items():
    #                 label_group_dict[label_name][metric].append(score)

    #     result = {'sample_count': {}}
    #     for attribute in label_group_dict.keys():
    #         for metric, scores in label_group_dict[attribute].items():
    #             mean_score = sum(scores) / len(scores)
    #             result.setdefault(metric, {})[attribute] = mean_score
    #             result['sample_count'][attribute] = len(scores)

    #     return MMEMetric.sort_nested_dict(result)

    @staticmethod
    def get_page_split(samples: List[Dict[str, Any]], page_info: Dict[str, Any]) -> Dict[str, Any]:
        result_list = defaultdict(list)
        for sample in samples:
            img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') else '_'.join(sample['img_id'].split('_')[:-1])
            page_info_s = page_info.get(img_name, {})
            if not sample.get('metric'):
                continue
            for metric, score in sample['metric'].items():
                gt = sample.get('norm_gt', sample['gt'])
                pred = sample.get('norm_pred', sample['pred'])
                result_list[metric].append({
                    'image_name': img_name,
                    'metric': metric,
                    'attribute': 'ALL',
                    'score': score,
                    'upper_len': max(len(gt), len(pred))
                })
                for k, v in page_info_s.items():
                    if isinstance(v, list):
                        for special_issue in v:
                            if 'table' not in special_issue:
                                result_list[metric].append({
                                    'image_name': img_name,
                                    'metric': metric,
                                    'attribute': special_issue,
                                    'score': score,
                                    'upper_len': max(len(gt), len(pred))
                                })
                    else:
                        result_list[metric].append({
                            'image_name': img_name,
                            'metric': metric,
                            'attribute': f"{k}: {v}",
                            'score': score,
                            'upper_len': max(len(gt), len(pred))
                        })

        result = {}
        if result_list.get('Edit_dist'):
            df = pd.DataFrame(result_list['Edit_dist'])
            up_total_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: (x["score"] * x['upper_len']).sum() / x['upper_len'].sum()).groupby('attribute').mean()
            result['Edit_dist'] = up_total_avg.to_dict()

        for metric in result_list.keys():
            if metric == 'Edit_dist':
                continue
            df = pd.DataFrame(result_list[metric])
            page_avg = df.groupby(["image_name", "attribute"])['score'].mean().groupby('attribute').mean()
            result[metric] = page_avg.to_dict()

        return MMEMetric.sort_nested_dict(result)


from collections import defaultdict
from typing import Optional, Dict, List, Any

from mmengine.evaluator import BaseMetric
from opencompass.registry import METRICS
import Levenshtein
import evaluate
import random
from .table_metric import TEDS
from utils.read_files import save_paired_result


@METRICS.register_module()
class CustomTEDSMetric(BaseMetric):
    """Custom TEDS metric for evaluating table structure similarity.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.teds = TEDS(structure_only=False)
        self.teds_structure_only = TEDS(structure_only=True)

    def process(self, data_batch: Dict, data_samples: List[Dict]) -> None:
        """Process one batch of data and predictions."""
        for sample in data_samples:
            gt = sample.get('norm_gt', sample['gt'])
            pred = sample.get('norm_pred', sample['pred'])
            score = self.teds.evaluate(pred, gt)
            score_structure_only = self.teds_structure_only.evaluate(pred, gt)
            
            result = {
                'img_path': sample['img_id'],
                'task': 'TableRecognition',  # Assuming all samples are for table recognition
                'score': score,
                'score_structure_only': score_structure_only
            }
            self.results.append(result)

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute the metrics from processed results."""
        group_scores = defaultdict(list)
        group_scores_structure_only = defaultdict(list)
        
        for result in results:
            group_scores['all'].append(result['score'])
            group_scores_structure_only['all'].append(result['score_structure_only'])

        result_dict = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result_dict[f'TEDS_{group_name}'] = sum(scores) / len(scores)
            else:
                result_dict[f'TEDS_{group_name}'] = 'NaN'
                
        for group_name, scores in group_scores_structure_only.items():
            if len(scores) > 0:
                result_dict[f'TEDS_structure_only_{group_name}'] = sum(scores) / len(scores)
            else:
                result_dict[f'TEDS_structure_only_{group_name}'] = 'NaN'

        return result_dict


@METRICS.register_module()
class CustomBLEUMetric(BaseMetric):
    """Custom BLEU metric for evaluating text translation quality."""

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, 1e8))

    def process(self, data_batch: Dict, data_samples: List[Dict]) -> None:
        for sample in data_samples:
            gt = sample.get('norm_gt', sample['gt'])
            pred = sample.get('norm_pred', sample['pred'])
            self.results.append({
                'img_path': sample['img_id'],
                'task': 'TextTranslation',
                'gt': gt,
                'pred': pred
            })

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        predictions, references = [], []
        for result in results:
            predictions.append(result['pred'])
            references.append([result['gt']])
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        return {'BLEU_all': bleu_results["bleu"]}


@METRICS.register_module()
class CustomMETEORMetric(BaseMetric):
    """Custom METEOR metric for evaluating text translation quality."""

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.meteor = evaluate.load('meteor', keep_in_memory=True, experiment_id=random.randint(1, 1e8))

    def process(self, data_batch: Dict, data_samples: List[Dict]) -> None:
        for sample in data_samples:
            gt = sample.get('norm_gt', sample['gt'])
            pred = sample.get('norm_pred', sample['pred'])
            self.results.append({
                'img_path': sample['img_id'],
                'task': 'TextTranslation',
                'gt': gt,
                'pred': pred
            })

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        predictions, references = [], []
        for result in results:
            predictions.append(result['pred'])
            references.append([result['gt']])
        meteor_results = self.meteor.compute(predictions=predictions, references=references)
        return {'METEOR_all': meteor_results['meteor']}


@METRICS.register_module()
class CustomEditDistMetric(BaseMetric):
    """Custom Edit Distance metric for evaluating string similarity."""

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_batch: Dict, data_samples: List[Dict]) -> None:
        for sample in data_samples:
            gt = sample.get('norm_gt', sample['gt'])
            pred = sample.get('norm_pred', sample['pred'])
            upper_len = max(len(pred), len(gt))
            edit_dist = Levenshtein.distance(pred, gt)
            result = {
                'img_path': sample['img_id'],
                'task': 'StringSimilarity',
                'edit_dist': edit_dist / upper_len if upper_len > 0 else 0,
                'edit_num': edit_dist
            }
            self.results.append(result)

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        edit_distances = [result['edit_dist'] for result in results]
        if edit_distances:
            avg_edit_dist = sum(edit_distances) / len(edit_distances)
        else:
            avg_edit_dist = 'NaN'
        return {'Edit_dist_all': avg_edit_dist}