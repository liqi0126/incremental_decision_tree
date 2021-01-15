
from model.utils import AttrType
import sys
sys.path.append('..')

datasets_config = {
    'forest': {
        'csv_path': './dataset/forest/covtype.csv',
        'yml_config': './experiments/yamls/forest.yml'
    },
    'skin': {
        'csv_path': './dataset/skin/skin.csv',
        'yml_config': './experiments/yamls/skin.yml'
    },
    'gas': {
        'csv_path': './dataset/gas/gas.csv',
        'yml_config': './experiments/yamls/gas.yml'
    },
    'poker': {
        'csv_path': './dataset/poker/poker.csv',
        'yml_config': './experiments/yamls/poker.yml',
        'attr_types': [AttrType.CATE] * 10
    },
    'activity_prediction': {
        'csv_path': './dataset/activity_prediction/activity_prediction.csv',
        'yml_config': './experiments/yamls/activity_prediction.yml',
    },
    'activity_recognition': {
        'csv_path': './dataset/activity_recognition/activity_recognition.csv',
        'yml_config': './experiments/yamls/activity_recognition.yml',
    },
    'moa2': {
        'csv_path': './dataset/moa/dataset1.csv',
        'yml_config': './experiments/yamls/moa.yml',
        'attr_types': [AttrType.CATE] * 5
    },
    'moa3': {
        'csv_path': './dataset/moa/dataset2.csv',
        'yml_config': './experiments/yamls/moa.yml',
        'attr_types': [AttrType.CATE] * 5
    },
    'moa4': {
        'csv_path': './dataset/moa/dataset3.csv',
        'yml_config': './experiments/yamls/moa.yml',
        'attr_types': [AttrType.CATE] * 5
    },
    'moa5': {
        'csv_path': './dataset/moa/dataset4.csv',
        'yml_config': './experiments/yamls/moa.yml',
        'attr_types': [AttrType.CATE] * 5
    }
}
