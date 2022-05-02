import gzip
import pickle
import collections
import csv
import glob
import json

dataset_dir = "../../DISAPERE/final_dataset/"

SUBSETS = "train dev test".split()
REVIEW_LABELS = "coarse fine asp pol".split()
REBUTTAL_LABELS = "coarse fine".split()
HEADER = "split   dataset filename        sentence1       sentence2       label"

datasets = collections.defaultdict(list)

Row = collections.namedtuple("Row", HEADER.split())

label_sets = collections.defaultdict(set)

rows = []
for subset in SUBSETS:
  for filename in glob.glob(dataset_dir + subset + "/*"):
    with open(filename, 'r') as f:
      example = json.load(f)
      for sentence in example["review_sentences"]:
        for label in REVIEW_LABELS:
          rows.append(
              Row(subset, label, "_", sentence["text"], sentence["text"],
                  sentence[label]))
      for sentence in example["rebuttal_sentences"]:
        for label in REBUTTAL_LABELS:
          rows.append(
              Row(subset, "reb_" + label, "_", sentence["text"],
                  sentence["text"], sentence[label]))

for row in rows:
  label_sets[row.dataset].add(row.label)

label_maps = collections.defaultdict(dict)
for dataset, labels in label_sets.items():
  label_maps[dataset] = {label: i for i, label in enumerate(sorted(labels))}

with open("data/label_map.pkl", 'wb') as f:
  pickle.dump(label_maps, f)

with open("data/all_DISAPERE.tsv", 'w') as f:
  writer = csv.DictWriter(f, fieldnames=HEADER.split(), delimiter='\t')
  writer.writeheader()
  for row in rows:
    writer.writerow(row._asdict())
