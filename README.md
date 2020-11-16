# Particle Physics Result Replication
## Checkpoint #1

#### Brief Description of Structure

##### config
###### data-params.json
This file includes the parameters for data extraction.

##### src/data
###### etl.py
This file contains the functions necessary for data extraction.

##### run.py
This file successfully extracts the data.

#### In order to run the file run.py, execute the following command:
##### python run.py data

## Checkpoint #2

#### Brief Description of Structure

##### config
###### data-params.json
This file includes the parameters for data extraction.
###### model-params.json
This file includes the parameters for building models.
###### compare.json
This file includes the parameters for making histograms and ROC to measure how well features discriminate.

##### notebooks
###### EDA.ipynb
Basic EDA notebook

##### src/data
###### etl.py
This file contains the functions necessary for data extraction.

##### src/analysis
###### compare.py
This file creates histograms and ROC to measure how well features discriminate.

##### src/model
###### model.py
This file builds the baseline models.

##### run.py
This file successfully runs the targets.

#### In order to run the file run.py, execute the following command:
##### python run.py data
