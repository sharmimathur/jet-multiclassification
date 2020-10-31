import sys
import json

from etl import get_features_labels

sys.path.insert(0, 'src/data')

def main(targets):
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        data = get_features_labels(**data_cfg)


    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
