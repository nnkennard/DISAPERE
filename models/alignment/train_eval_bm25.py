import collections
import glob
import json
from rank_bm25 import BM25Okapi
import random
import stanza
import tqdm
import numpy as np
random.seed(131)

from rank_metrics import reciprocal_rank, average_precision

with open("nltk_stopwords.json", "r") as f:
  STOPWORDS = json.load(f)

STANZA_PIPELINE = stanza.Pipeline("en",
                                  processors="tokenize,lemma",
                                  tokenize_no_ssplit=True)


def stanza_preprocess(sentences):
  doc = STANZA_PIPELINE("\n\n".join(sentences))
  lemmatized = []
  for sentence in doc.sentences:
    sentence_lemmas = []
    for token in sentence.tokens:
      (token_dict,) = token.to_dict()
      maybe_lemma = token_dict["lemma"].lower()
      if maybe_lemma not in STOPWORDS:
        sentence_lemmas.append(maybe_lemma)
    lemmatized.append(sentence_lemmas)
  return lemmatized

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



Example = collections.namedtuple("Example",
    "review_sentences rebuttal_sentences alignment")

def get_alignment(rebuttal_sentences):
  alignment = []
  for reb_i, reb_sentence in enumerate(rebuttal_sentences):
    _, aligned_indices = reb_sentence['alignment']
    if aligned_indices is None:
      alignment.append([])
    else:
      alignment.append(aligned_indices)
  assert len(alignment) == len(rebuttal_sentences)
  return alignment


def main():

  examples = collections.defaultdict(list)

  for subset in "train dev test".split():
    for filename in glob.glob(f'../../DISAPERE/final_dataset/{subset}/*'):
      print(filename)
      with open(filename, 'r') as f:
        obj = json.load(f)
        examples[subset].append(Example(
          stanza_preprocess([x['text'] for x in obj['review_sentences']]),
          stanza_preprocess([x['text'] for x in obj['rebuttal_sentences']]),
          get_alignment(obj['rebuttal_sentences'])
        ))

  rrs = []
  aps = []
  for train_example in tqdm.tqdm(examples['train']):
    model = BM25Okapi(train_example.review_sentences)
    for i, query in enumerate(train_example.rebuttal_sentences):
      aligned = train_example.alignment[i]
      if not aligned:
        continue
      scores = model.get_scores(query)
      is_relevant = convert_scores_to_is_relevant(scores, aligned)
      rrs.append(reciprocal_rank(is_relevant))
      aps.append(average_precision(is_relevant))

  print(np.mean(rrs))
  print(np.mean(aps))

      



if __name__ == "__main__":
  main()

