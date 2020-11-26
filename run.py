import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')

from etl import get_features_labels
import compare
from model import create_models
import test_run

def main(targets):
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        data = get_features_labels(**data_cfg)
        
    if 'compare' in targets:
        compare()
        
    if 'conv1d' in targets:
        create_models()

    if 'test' in targets:
        test_run()

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
