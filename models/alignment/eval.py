import collections
import json
import numpy as np

from rank_metrics import reciprocal_rank, average_precision
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


def get_correct_indices(discrete_map):
  correct_indices = {}
  for reb_i, reb_row in enumerate(discrete_map):
    correct_indices[reb_i] = [i for i, x in enumerate(reb_row[:-1]) if x]
  return correct_indices

def convert_scores_to_is_relevant(scores, correct_indices):
  is_relevant_map = [None] * len(scores)
  ordered_scores = list(reversed(sorted((score, i) for i, score in
  enumerate(scores))))
  for rank, (score, i) in enumerate(ordered_scores):
    if i in correct_indices:
      is_relevant_map[rank] = 1
    else:
      is_relevant_map[rank] = 0
  assert set(is_relevant_map) == set([0,1])
  return is_relevant_map

def evaluate(model, test_file):
  examples = []
  with open(test_file, 'r') as f:
    for line in f:
      examples.append(json.loads(line))
  rrs = []
  aps = []
  no_match_counter = []
  for example in examples:
    review_embs = model.encode(example['review_sentences'])
    rebuttal_embs = model.encode(example['rebuttal_sentences'])
    correct_indices = example['alignment']
    #print(correct_indices)
    cosine_scores = util.pytorch_cos_sim(rebuttal_embs, review_embs[:-1])
    other_cosine_scores = util.pytorch_cos_sim(rebuttal_embs, review_embs[:-1])
    #print(cosine_scores)
    for reb_i, reb_scores in enumerate(cosine_scores):
      no_match_counter.append((len(correct_indices[reb_i])==0,
      (other_cosine_scores[reb_i][-1] == max(other_cosine_scores[reb_i])).item()))
      if not correct_indices[reb_i]:
        continue
      else:
        is_relevant_map = convert_scores_to_is_relevant(
        reb_scores, correct_indices[reb_i])
        aps.append(average_precision(is_relevant_map))
        rrs.append(reciprocal_rank(is_relevant_map))
  print(f"MAP: {np.mean(aps)}, MRR: {np.mean(rrs)}")


def main():
  model = SentenceTransformer('outputs/best_model.bin')
  data_dir = "prepared_data/"
  for subset in "train dev test".split():
    print(f"{subset}\t", end="")
    evaluate(model, f"{data_dir}/{subset}.jsonl")


if __name__ == "__main__":
  main()

