import json
import os
from typing import List, Dict
from torch.utils.data import Dataset
from opencompass.registry import DATASETS

@DATASETS.register_module()
class OmnidocbenchDataset(Dataset):
    """Custom dataset for loading images and corresponding ground truth from JSON files.
    
    Args:
        data_dir (str): The path to the dataset directory.
        pipeline (List[dict]): A list of dictionaries describing the data augmentation pipeline.
    """
    
    def __init__(self, data_dir: str, pipeline: List[dict]) -> None:
        self.pipeline = Compose(pipeline)
        self.load_data(data_dir)

    def load_data(self, data_dir: str) -> None:
        """Loads image paths and their corresponding ground truths."""
        self.data_list = []
        gt_file = os.path.join(data_dir, 'OmniDocBench.json')  # Assuming a single JSON file contains all annotations
        
        with open(gt_file, 'r') as f:
            gt_samples = json.load(f)
        
        for sample in gt_samples:
            img_name = os.path.basename(sample["page_info"]["image_path"])
            img_path = os.path.join(data_dir, 'images', img_name)
            
            if not os.path.exists(img_path):
                print(f'Warning: Image {img_name} does not exist.')
                continue
            
            self.data_list.append({
                'img_path': img_path,
                'gt': sample,  # Store the entire ground truth dictionary or relevant parts
            })

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        """Gets an item from the dataset at the specified index after applying the pipeline."""
        data_sample = self.data_list[idx]
        data_sample = self.pipeline(data_sample)
        return data_sample
    
import json
import os
from collections import defaultdict
from typing import List, Dict, Any
from torch.utils.data import Dataset
from mmengine.dataset import Compose
from opencompass.registry import DATASETS
from tqdm import tqdm
from loguru import logger
import Levenshtein

@DATASETS.register_module()
class OmnidocbenchDataset_Filter(Dataset):
    """Custom dataset for loading images and corresponding ground truth from JSON files.
    
    Args:
        data_dir (str): The path to the dataset directory.
        pipeline (List[dict]): A list of dictionaries describing the data augmentation pipeline.
        match_method (str): Method used for matching predictions to ground truth.
        filtered_types (Dict[str, Any], optional): Filters to apply on page attributes.
    """
    
    def __init__(self, data_dir: str, pipeline: List[dict], match_method: str = 'quick_match', filtered_types: Dict[str, Any] = None) -> None:
        self.pipeline = Compose(pipeline)
        self.match_method = match_method
        self.filtered_types = filtered_types or {}
        
        gt_path = os.path.join(data_dir, 'OmniDocBench.json')  # Assuming a single JSON file contains all annotations
        pred_folder = os.path.join(data_dir, 'predictions')
        
        with open(gt_path, 'r') as f:
            gt_samples = json.load(f)

        filtered_gt_samples = []
        if self.filtered_types:
            for gt_sample in gt_samples:
                select_flag = True
                for k, v in self.filtered_types.items():
                    if gt_sample["page_info"]["page_attribute"].get(k) != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = gt_samples

        self.samples = self.get_matched_elements(filtered_gt_samples, pred_folder)
        
    def load_data(self, gt_samples: List[Dict], pred_folder: str) -> Dict[str, List]:
        matched_samples_all = {
            'text_block': [],
            'display_formula': [],
            'table': [],
            'reading_order': []
        }

        process_bar = tqdm(gt_samples, ascii=True, ncols=140)
        for sample in process_bar:
            img_name = os.path.basename(sample["page_info"]["image_path"])
            pred_path = self.find_prediction_file(pred_folder, img_name)
            
            if not pred_path:
                logger.warning(f'No prediction for {img_name}')
                continue

            process_bar.set_description(f'Processing {os.path.basename(pred_path)}')
            pred_content = read_md_file(pred_path)
            result = self.process_get_matched_elements(sample, pred_content, img_name)

            for key, value in result.items():
                if value:
                    matched_samples_all[key].extend(value)

        return matched_samples_all
    
    def find_prediction_file(self, pred_folder: str, img_name: str) -> str:
        base_name = img_name.rsplit('.', 1)[0]
        extensions = ['.md', '.mmd']
        for ext in extensions:
            pred_path = os.path.join(pred_folder, base_name + ext)
            if os.path.exists(pred_path):
                return pred_path
        return ''
    
    def get_page_elements(self, selected_annos: Dict) -> Dict[str, List]:
        pass
    
    def get_sorted_text_list(self, selected_annos: List[Dict]) -> List[Dict]:
        pass
    
    def get_order_paired(self, order_match_s: List[Dict], img_name: str) -> Dict:
        pass
    
    def formula_format(self, formula_matches: List[Dict], img_name: str) -> List[Dict]:
        pass
    
    def process_get_matched_elements(self, sample: Dict, pred_content: str, img_name: str) -> Dict[str, List]:
        pass

    def get_matched_elements(self, gt_samples: List[Dict], pred_folder: str) -> Dict[str, List]:
        return self.load_data(gt_samples, pred_folder)

    def __len__(self) -> int:
        return sum(len(v) for v in self.samples.values())

    def __getitem__(self, idx: int) -> Dict:
        category = next((cat for cat, samples in self.samples.items() if idx < len(samples)), None)
        if category is None:
            raise IndexError('Index out of range.')
        data_sample = self.samples[category][idx]
        data_sample = self.pipeline(data_sample)
        return data_sample