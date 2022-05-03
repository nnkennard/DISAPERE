import argparse
import json
import os

import data_prep_lib as dpl

parser = argparse.ArgumentParser(
    description="Convert the filtered database entries into a cleaned dataset.")
parser.add_argument('-i',
                    '--intermediate_file',
                    default="filtered_database.json",
                    type=str,
                    help='path to text dump from annotation server')


def read_filtered_dataset(intermediate_file):
  with open(intermediate_file, 'r') as f:
    obj = json.load(f)
    return obj["annotations"], obj["text"]

def process_review_sentences(review_sentence_annotations, review_text,
  merge_prev):

  merge_map = {i:set([i]) for i in range(len(review_text))}

  # If a sentence is merged with the previous sentence, copy the annotation
  # over
  for i in range(len(review_text)):
    str_index = str(i)
    if str_index not in review_sentence_annotations:
      review_sentence_annotations[str_index] = review_sentence_annotations[str(i-1)]
      new_indices = merge_map[i].union(merge_map[i-1])
      merge_map[i] = new_indices
      merge_map[i-1] = new_indices
  assert len(review_sentence_annotations) == len(review_text)

  final_sentence_list = []
  for i, (sentence_text_info, merge_prev_val) in enumerate(zip(review_text,
  merge_prev)):
    sentence_text = sentence_text_info["text"]
    suffix = sentence_text_info["suffix"]
    relevant = review_sentence_annotations[str(i)]
    assert i == relevant["review_sentence_index"] or i == relevant["review_sentence_index"] + 1
    coarse, fine, asp, pol = dpl.clean_review_label(relevant)
    final_sentence_list.append(
        dpl.ReviewSentence(relevant["review_id"], i,
                           sentence_text, suffix, coarse, fine, asp, pol))
  return final_sentence_list, merge_map


def process_rebuttal_sentences(rebuttal_sentence_annotations, rebuttal_text,
    merge_map):

  assert [k==v for k,v in enumerate(merge_map)]
  final_rebuttal_sentences = []

  (review_id, rebuttal_id), = set([
      (dpl.get_fields(i)["review_id"], dpl.get_fields(i)["rebuttal_id"])
      for i in rebuttal_sentence_annotations
  ])

  for sentence in rebuttal_sentence_annotations:
    index, label, coarse, alignment, details = dpl.clean_rebuttal_label(
        sentence, merge_map)

    final_rebuttal_sentences.append(
        dpl.RebuttalSentence(review_id, rebuttal_id, index,
                             rebuttal_text[index]["text"],
                             rebuttal_text[index]["suffix"],
                             coarse, label, alignment, details))
    prev_sentence = sentence


  return final_rebuttal_sentences


def process_annotation(annotation, text):
  metadata = dict(text[dpl.METADATA]) # Get text metadata
  metadata["annotator"] = annotation["annotator"]

  # Which review sentences are supposed to be merged. We don't actually use
  # this, but it's sometimes relevant for matching alignments.
  merge_prev = json.loads(
      dpl.get_fields(annotation["review_annotation"])["errors"])["merge_prev"]

  processed_review, merge_map = process_review_sentences(
      annotation["review_sentences"],text["review"], merge_prev)
  if processed_review is not None:
    processed_rebuttal = process_rebuttal_sentences(
    annotation["rebuttal_sentences"], text["rebuttal"], merge_map)
    return dpl.Annotation(processed_review, processed_rebuttal, metadata)
  else:
    return None


def process_all_annotations(annotation_collections, text_map):
  final_annotations = []
  extra_annotations = []
  for review_id, annotations in annotation_collections.items():
    if review_id == "example_id": # Skip placeholder example
      continue

    # Sort annotators in order of reliability
    annotators = sorted(annotations.keys(),
                        key=lambda x: dpl.preferred_annotators.index(x))
    # anno0 is the adjudicator, and should be the preferred annotator if
    # available
    assert annotators[0] == "anno0" or "anno0" not in annotators

    # Try to process annotatiosn
    maybe_valid_annotations = [
        process_annotation(annotations[annotator], text_map[review_id])
        for annotator in annotators
        if annotations[annotator] is not None
    ]

    # Retain valid annotations
    valid_annotations = [
        annotation for annotation in maybe_valid_annotations if annotation is not None
    ]

    if valid_annotations:
      # Annotation from most reliable annotator goes into the final dataset
      final_annotations.append(valid_annotations.pop(0))
      # All other valid annotations are 'extra'
      extra_annotations += valid_annotations

  return final_annotations, extra_annotations

def write_annotations_to_dir(annotations, dir_name, append_annotator=False):
  for subdir in ["", "train", "dev", "test"]:
    subdir_name = dir_name + "/" + subdir
    if not os.path.exists(subdir_name):
      os.makedirs(subdir_name)

  for annotation in annotations:
    if annotation is None:
      continue
    annotation.write_to_dir(dir_name, append_annotator)



def main():

  args = parser.parse_args()

  annotation_collections, text_map = read_filtered_dataset(
      args.intermediate_file)

  final_annotations, extra_annotations = process_all_annotations(
      annotation_collections, text_map)

  rev_lens = []
  reb_lens = []
  for i in final_annotations:
    rev_lens.append(len(i.review_sentences))
    reb_lens.append(len(i.rebuttal_sentences))


  write_annotations_to_dir(final_annotations, "../../temp_DISAPERE/final_dataset/")
  write_annotations_to_dir(extra_annotations,
                             "../../temp_DISAPERE/extra_annotations/",
                             append_annotator=True)


if __name__ == "__main__":
  main()
