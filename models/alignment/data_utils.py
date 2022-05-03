import glob
import json
import os
import random
from tqdm import tqdm
import numpy as np



NO_MATCH = "NO_MATCH"
NUM_NEG_SAMPLES = 2


def read_data(data_dir):
  sentence_pairs = []
  print("Loading data from " + data_dir)
  for filename in tqdm(list(glob.glob(data_dir + "/*.json"))):
    with open(filename, 'r') as f:
      pair = json.load(f)
      review_texts = [x["text"] for x in pair["review_sentences"]]
      for reb_i, rebuttal_sent in enumerate(pair["rebuttal_sentences"]):
        align_type, align_indices = rebuttal_sent["alignment"]
        if align_indices is None:
          sentence_pairs.append((rebuttal_sent["text"], NO_MATCH, 1))
          neg_candidate_indices = list(range(len(pair["review_sentences"])))
        else:
          for align_idx in align_indices:
            sentence_pairs.append(
                (rebuttal_sent["text"], review_texts[align_idx], 1))
          neg_candidate_indices = list(
              sorted(set(range(len(review_texts))) - set(align_indices)))

        if len(neg_candidate_indices) < NUM_NEG_SAMPLES:
          selected_neg_indices = neg_candidate_indices
        else:
          selected_neg_indices = random.sample(neg_candidate_indices,
          NUM_NEG_SAMPLES)
        for negative_ind in selected_neg_indices:
          sentence_pairs.append(
              (rebuttal_sent["text"], review_texts[negative_ind], 0))
  return sentence_pairs
