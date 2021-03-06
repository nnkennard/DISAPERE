import collections
import glob
import json
import os
from tqdm import tqdm
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
from rank_metrics import reciprocal_rank, average_precision


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

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


def calculate_official_mrr(cosine_scores, rebuttal_sentences):
  reb_len, rev_len = cosine_scores.shape
  rrs = []
  aps = []
  no_match_rrs = []
  no_match_aps = []
  assert reb_len == len(rebuttal_sentences)
  for reb_i, rebuttal_sentence in enumerate(rebuttal_sentences):
    _, aligned_indices = rebuttal_sentence['alignment']
    if aligned_indices is None:
      relevant = [rev_len - 1]
    else:
      relevant = aligned_indices
    is_relevant = convert_scores_to_is_relevant(cosine_scores[reb_i],
    relevant)
    if aligned_indices is None:
      no_match_rrs.append(reciprocal_rank(is_relevant))
      no_match_aps.append(average_precision(is_relevant))
    else:
      rrs.append(reciprocal_rank(is_relevant))
      aps.append(average_precision(is_relevant))
  return rrs, aps, no_match_rrs, no_match_aps



def eval_dir(model_save_path,
             data_dir="../../DISAPERE/final_dataset",
             subset="dev"):

  glob_path = data_dir + "/" + subset + "/*.json"
  logger.info("Loading model...")
  model = SentenceTransformer(model_save_path)

  no_match_ranks = collections.defaultdict(list)

  all_rr = []
  official_rr = []
  official_ap = []
  no_match_official_rr = []
  no_match_official_ap = []
  for filename in glob.glob(glob_path):
    with open(filename) as fin:
      data = json.load(fin)
    rebuttal_sentences_text = [t["text"] for t in data["rebuttal_sentences"]]
    review_sentences_text = [t["text"] for t in data["review_sentences"]
                            ] + ["NO_MATCH"]
    rebuttal_sentences_emb = model.encode(rebuttal_sentences_text)
    review_sentences_emb = model.encode(review_sentences_text)
    # Why does NO_MATCH not get encoded?
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(rebuttal_sentences_emb,
                                         review_sentences_emb)
    rrs, aps, no_match_rrs, no_match_aps = calculate_official_mrr(cosine_scores,
    data['rebuttal_sentences'])
    official_ap += aps
    official_rr += rrs
    no_match_official_ap += no_match_aps
    no_match_official_rr += no_match_rrs
    ranks = torch.argsort(-cosine_scores, dim=1)
    for ctr, rb_s in enumerate(data["rebuttal_sentences"]):
      rr = 0
      ranks_rbs = ranks[ctr]
      no_match_rank = ranks_rbs.tolist().index(len(ranks_rbs) - 1)
      _, alignments = rb_s["alignment"]
      if alignments is None:
        alignments = [len(review_sentences_text) - 1]
        no_match_ranks["non_aligned"].append(no_match_rank)
      else:
        no_match_ranks["aligned"].append(no_match_rank)
      success_ctr = 0  # number of alignments found successfully so far
      for r_ctr, r in enumerate(ranks_rbs):
        if r in alignments:
          rr += (1 / (r_ctr + 1 - success_ctr))
          success_ctr += 1
      rr = rr / len(alignments)
      all_rr.append(rr)
  with open("results.txt", "a") as f:
    f.write("{} {} Old MRR: {}\n".format(model_save_path, subset, np.mean(all_rr)))
    f.write("{} {} MAP: {}\n".format(model_save_path, subset,
    np.mean(official_ap + no_match_official_ap)))
    f.write("{} {} MRR: {}\n".format(model_save_path, subset,
    np.mean(official_rr+no_match_official_rr)))
    f.write("{} {} Easy MAP: {}\n".format(model_save_path, subset,
    np.mean(official_ap)))
    f.write("{} {} Easy MRR: {}\n".format(model_save_path, subset,
    np.mean(official_rr)))
    f.write("{} {} No match MAP: {}\n".format(model_save_path, subset,
    np.mean(no_match_official_ap)))
    f.write("{} {} No match MRR: {}\n".format(model_save_path, subset,
    np.mean(no_match_official_rr)))


    f.write("=" *80 +"\n")
    with open("no_match_rank_" + subset + ".json", 'w') as f:
      json.dump(no_match_ranks, f)


if __name__ == "__main__":
  #NAACL_FINAL_MODEL = "output/training_OnlineConstrativeLoss-2022-01-15_12-53-37"
  NAACL_FINAL_MODEL ="output/training_OnlineConstrativeLoss-2022-05-03_13-04-08"
  #eval_dir(NAACL_FINAL_MODEL,
  #         data_dir="../../../DISAPERE/final_dataset",
  #         subset="dev")
  eval_dir(NAACL_FINAL_MODEL,
           data_dir="../../DISAPERE/final_dataset",
           subset="dev")

