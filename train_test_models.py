import sys
from training import train_models_basic, train_models_train_size
from training import train_models_permute_labels, train_small_models_permute_labels
from training import train_models_norm_comps, train_gnb_one_feature
from training import train_test_chbmit

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception(f'--> Input requires only one parameter, got {len(sys.argv)} instead.')

    if sys.argv[1] == 'per_file':
        train_models_basic.run_models(norm_type='per_file')

    if sys.argv[1] == 'all_file':
        train_models_basic.run_models(norm_type='all_file')

    if sys.argv[1] == 'train_size':
        train_models_train_size.run_models(norm_type='per_file')

    if sys.argv[1] == 'permute_labels':
        train_models_permute_labels.run_models(norm_type='per_file')

    if sys.argv[1] == 'small_models_permute':
        train_small_models_permute_labels.run_models(norm_type='per_file')

    if sys.argv[1] == 'gnb_one_feature':
        train_gnb_one_feature.run_models()

    if sys.argv[1] == 'norm_comps':
        train_models_norm_comps.run_models()

    if sys.argv[1] == 'chb_mit':
        train_test_chbmit.run_models()
