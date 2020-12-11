import sys
import json
import yaml

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')

from etl import get_features_labels
from compare import compare
from model import create_models
from baseline_model import create_baseline_model

def main(targets):
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        data = get_features_labels(**data_cfg)
        
    if 'compare' in targets:
        compare('config/data-params.json')
        
    if 'conv1d' in targets:
        with open('config/data-params.yml') as file:
            # The FullLoader parameter handles the conversion from YAML
#                 # scalar values to Python the dictionary format
            definitions = yaml.load(file, Loader=yaml.FullLoader)

                
        with open('config/model-params.json') as fh:
            data_cfg = json.load(fh)
        create_models(**definitions, **data_cfg)
        #create_models('config/data-params.yml','config/model-params.json')
        
    if 'test' in targets:
        with open('config/test-data-params.yml') as file:
            # The FullLoader parameter handles the conversion from YAML
#                 # scalar values to Python the dictionary format
            definitions = yaml.load(file, Loader=yaml.FullLoader)

                
        with open('config/model-params.json') as fh:
            data_cfg = json.load(fh)
        create_baseline_model(**definitions, **data_cfg)
        #compare('config/test-data-params.json')
        #create_baseline_model('config/test-data-params.yml','config/model-params.json')

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
