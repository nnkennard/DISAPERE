import json
import os

TRAIN, EVAL, PREDICT = "train eval predict".split()
DEV, TEST = "dev test".split()
MODES = [TRAIN, EVAL, PREDICT]
SUBSETS = [TRAIN, DEV, TEST]

REVIEW_KEYS = ["review_action", "fine_review_action", "aspect", "polarity"]
REBUTTAL_KEYS = ["rebuttal_action", "rebuttal_stance"]


def get_features_and_labels(data_dir, get_labels=False):
  features = []
  texts = []
  identifiers = []
  with open(f"{data_dir}/features.jsonl", "r") as f:
    for line in f:
      example = json.loads(line)
      features.append(example["features"])
      texts.append(example["text"])
      identifiers.append(example["identifier"])
  if get_labels:
    labels = []
    with open(f"{data_dir}/labels.jsonl", "r") as f:
      for i, line in enumerate(f):
        example = json.loads(line)
        assert example["identifier"] == identifiers[i]
        labels.append(example["label"])
  else:
    labels = None

  return identifiers, features, texts, labels


def get_examples(filename):
  with open(filename, 'r') as f:
    example_dicts = [json.loads(l.strip()) for l in f.readlines()]

  joint_list = [[example[k] for k in "text index target_index".split()]
               for example in example_dicts]
  return zip(*joint_list)
