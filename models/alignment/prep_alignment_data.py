import collections
import glob
import json
import numpy as np
import os

NO_MATCH = "NO_MATCH"

def get_review_info(filename):
  with open(filename, 'r') as f:
    obj = json.load(f)
    review_sentences = [x["text"] for x in obj["review_sentences"]] + [NO_MATCH]
    rebuttal_sentences = [x["text"] for x in obj["rebuttal_sentences"]]
    alignment = np.zeros([len(rebuttal_sentences), len(review_sentences)])
    for reb_i, rebuttal_sentence in enumerate(obj["rebuttal_sentences"]):
      _, aligned_review_sentences = rebuttal_sentence['alignment']
      if aligned_review_sentences is None:
        alignment[reb_i][-1] = 1
      else:
        for rev_i in aligned_review_sentences:
          alignment[reb_i][rev_i] = 1
  return review_sentences, rebuttal_sentences, alignment


def main():
  examples = collections.defaultdict(list)
  os.makedirs('prepared_data/', exist_ok=True)
  for subset in "train dev test".split():
    with open(f'prepared_data/{subset}.jsonl', 'w') as f:
      for filename in glob.glob(f"../../DISAPERE/final_dataset/{subset}/*"):
        review_sentences, rebuttal_sentences, alignment = get_review_info(filename)
        f.write(json.dumps({
                  "rebuttal_sentences": rebuttal_sentences,
                  "review_sentences": review_sentences,
                  "alignment": alignment.tolist(),
                })+"\n")


if __name__ == "__main__":
  main()

