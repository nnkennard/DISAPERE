from .models import *

import collections
import json
import yaml

color_map = {
  "Accept": "green1",
  "Reject": "yellow",
  "Maybe-arg": "red1",
  "Non-arg": "purple1",
  "Error": "blue1"
}


def format_labels():
  with open("orda/orda_data/labels.yaml", 'r') as f:
    LABELS = yaml.safe_load(f)
  label_map = collections.defaultdict(list)

  for obj in LABELS["review_categories"]:
    label_map[obj["short"]] = ["-- " + obj["name"]] + obj["subcategories"]

  allowed_menus = collections.defaultdict(dict)
  for obj in LABELS["allowed_menus"]:
    if obj["allowed"] is not None:
      allowed_menus[obj["name"]]["allowed"] = obj["allowed"]
    else:
      allowed_menus[obj["name"]]["allowed"] = []
    if obj["required"] is not None:
      allowed_menus[obj["name"]]["required"] = obj["required"]
    else:
      allowed_menus[obj["name"]]["required"] = []

  rebuttal_relation_map = collections.defaultdict(list)
  for obj in LABELS["rebuttal_relations"]:
    rebuttal_relation_map[obj["category"]].append(
        (obj["key"], obj["description"]))

  rebuttal_relations = []
  for category in "Accept Reject Maybe-arg Non-arg Error".split():
    for key, desc in rebuttal_relation_map[category]:
      rebuttal_relations.append((category, color_map[category], key, desc))

  return label_map, allowed_menus, rebuttal_relations, LABELS["alignment_categories"]


FORMATTED_LABELS, ALLOWED_MENUS, REBUTTAL_RELATIONS, ALIGNMENT_CATEGORIES = format_labels()


def get_latest_review_annotation(example, initials):

  latest_overall = ReviewAnnotation.objects.filter(
      review_id=example.review_id, initials=initials).order_by("-id")[0]
  if not latest_overall.is_valid:
    return None
  merge_prev = json.loads(latest_overall.errors.replace("'", '"'))["merge_prev"]
  sentence_annotations = ReviewSentenceAnnotation.objects.filter(
      review_id=example.review_id, initials=initials)
  max_sentence_index = len(Sentence.objects.filter(comment_id=example.review_id))
  labels = []
  for i in range(max_sentence_index):
    print("Looking for old annotation ", initials, i)
    if merge_prev[i]:
      labels.append(labels[-1])
    else:
      labels.append(
          json.loads(
              ReviewSentenceAnnotation.objects.filter(
                  review_id=example.review_id,
                  initials=initials,
                  review_sentence_index=i).order_by("-id")[0].labels))
  return labels


def get_latest_rebuttal_annotation(example, initials):

  this_annotator_annotations = RebuttalSentenceAnnotation.objects.filter(
      review_id=example.review_id, initials=initials)
  latest_annotations = []

  for i in range(example.num_rebuttal_sentences):
    maybe_annotations = RebuttalSentenceAnnotation.objects.filter(
        review_id=example.review_id,
        initials=initials,
        rebuttal_sentence_index=i).order_by("-id")
    if not maybe_annotations:
      latest_annotations.append((None, None))
    else:
      latest_annotations.append((maybe_annotations[0].aligned_review_sentences,
                                maybe_annotations[0].relation_label))
  return latest_annotations

def make_presentation_example(assignment):
  return {
      "reviewer":
          assignment.example.reviewer,
      "review_status":
          assignment.is_review_complete,
      "rebuttal_status":
          " / ".join([
              str(assignment.num_completed_sentences),
              str(assignment.num_rebuttal_sentences),
          ]),
      "title":
          assignment.example.title,
      "rebuttal_id":
          assignment.example.rebuttal_id,
      "review_id":
          assignment.example.review_id,
  }


def get_this_annotator_assignments(initials, batch_size=30):

  all_assignments = AnnotatorAssignment.objects.filter(
      initials=initials).order_by("id")
  assignments_to_show = []
  incomplete_counter = 0
  previously_completed_counter = 0
  for a in all_assignments:
    if a.num_rebuttal_sentences > a.num_completed_sentences:
      incomplete_counter += 1
      assignments_to_show.append(a)
    else:
      assert a.num_completed_sentences == a.num_rebuttal_sentences
      if False:  # Actually, if completed over 24 hours ago
        previously_completed_counter += 1
      else:
        assignments_to_show.append(a)
    if incomplete_counter == batch_size:
      break

  return previously_completed_counter, assignments_to_show


def get_htmlified_sentences(supernote_id):
  sentences = Sentence.objects.filter(comment_id=supernote_id)
  final_sentences = []
  for i, sentence in enumerate(sentences):
    final_sentences.append({"text": sentence.text + sentence.suffix, "idx": i})
  return final_sentences


def get_annotation_context(initials, review_id, index):
  example = Example.objects.get(review_id=review_id)
  rebuttal_id = example.rebuttal_id
  review_annotations = get_latest_review_annotation(example, initials)
  rebuttal_sentences = get_htmlified_sentences(rebuttal_id)
  rebuttal_annotations = get_latest_rebuttal_annotation(example, initials)
  statuses = [x is not None for x, _ in rebuttal_annotations]
  previous_alignment, previous_label = rebuttal_annotations[index -1]


  return {
      "text": {
          "review_sentences": get_htmlified_sentences(review_id),
          "rebuttal_sentences": rebuttal_sentences,
          "rebuttal_sentence": rebuttal_sentences[index],
      },
      "other_annotations": {
          "statuses": statuses,
          "previous_label": previous_label,
          "previous_alignment": previous_alignment,
          "review_annotations": review_annotations
      }
  }
