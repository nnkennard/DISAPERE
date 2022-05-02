# DISAPERE: A Dataset for DIscourse Structure in Academic PEer REview

## Downloading the DISAPERE dataset

To download the dataset, run

```
wget ???
```

## Other information

Besides the DISAPERE dataset, this repository contains code for:
1. Running the DISAPERE annotation server
2. Processing the output of the DISAPERE annotation server
3. Training the classification model on the DISAPERE dataset (Section ?? in the paper)
4. Training the alignment models on the DISAPERE dataset (Section ?? in the paper)
5. Producing the analysis plots from Section ?? in the paper
6. Running the DISAPERE browser

We also include raw annotations from which DISAPERE was produced (anonymized)

### Setup

```
conda create --name disapere_env python=3.8
conda activate disapere_env
python -m pip install -r data_requirements.txt
```

### Running the DISAPERE annotation server

### Processing the output of the DISAPERE annotation server

```
python handle_database.py -a <annotation_file> -t <text_file>
python clean_examples.py
```
This produces the DISAPERE dataset from the original server output. For new server output, it might need some changes to `data/prep/subset_map.json`.

### Training the classification model

### Training the alignment models

### Running the DISAPERE browser
