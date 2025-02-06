#### ----------------------------  Imports  ------------------------------ ####
import sys
from training import (
    train_models_basic,
    train_models_train_size,
    train_models_permute_labels,
    train_small_models_permute_labels,
    train_models_norm_comps,
    train_gnb_one_feature,
    train_test_chbmit
)
from sz_utils import post_processing, time_plots

#### ---------------------------  Task Mapping  --------------------------- ####
TASKS = {
    'per_file': lambda: train_models_basic.run_models(norm_type='per_file'),
    'all_file': lambda: train_models_basic.run_models(norm_type='all_file'),
    'time_plots': time_plots.run_time_plots,
    'post_processing': post_processing.run_processing,
    'train_size': lambda: train_models_train_size.run_models(norm_type='per_file'),
    'permute_labels': lambda: train_models_permute_labels.run_models(norm_type='per_file'),
    'small_models_permute': lambda: train_small_models_permute_labels.run_models(norm_type='per_file'),
    'gnb_one_feature': train_gnb_one_feature.run_models,
    'norm_comps_mouse': train_models_norm_comps.run_models,
    'norm_comps_chb_mit': train_test_chbmit.run_models,
}

#### ----------------------------  Main Script  ---------------------------- ####
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError(f'Expected one argument, got {len(sys.argv) - 1}. Available tasks: {", ".join(TASKS.keys())}')

    task = sys.argv[1]
    if task not in TASKS:
        raise ValueError(f'Invalid task "{task}". Available tasks: {", ".join(TASKS.keys())}')

    print(f'--> Running task: {task}')
    TASKS[task]()