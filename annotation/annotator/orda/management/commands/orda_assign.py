from django.core.management.base import BaseCommand
from orda.models import *

import json
from tqdm import tqdm


def annotator_gen():
  annotators = Annotator.objects.filter(is_staff=False)
  annotator_pairs = []
  for i, ann1 in enumerate(annotators):
    for ann2 in annotators[i + 1:]:
      annotator_pairs.append((ann1.initials, ann2.initials))

  index = 0
  while True:
    yield annotator_pairs[index % len(annotator_pairs)]
    index += 1


class Command(BaseCommand):
  args = '<foo bar ...>'
  help = 'Given a json file, update database for alignments'

  def handle(self, *args, **options):

    result = input(("Are you sure you want to delete all annotator "
                    "assignments and repopulate the database? Y/N >"))
    if not result == "Y":
      return

    for ann in Annotator.objects.filter(is_staff=True):
        for i in range(1, 4):
            for example in Example.objects.filter(interleaved_index=i):
                assignment = AnnotatorAssignment(
                  example=example,
                  initials=ann.initials,
                  num_rebuttal_sentences=example.num_rebuttal_sentences,
                  num_completed_sentences=0, is_review_complete=False,
                  is_review_valid=False)
                assignment.save()

    return



    #AnnotatorAssignment.objects.all().delete()
    for ann in Annotator.objects.filter():
       assignment = AnnotatorAssignment(
                  example=Example.objects.get(interleaved_index=0),
                  initials=ann.initials,
                  num_rebuttal_sentences=3,
                  num_completed_sentences=0, is_review_complete=False,
                  is_review_valid=False)
       assignment.save()

    ann_gen = annotator_gen()
    max_interleaved_index = max(
        i["interleaved_index"]
        for i in Example.objects.all().values("interleaved_index"))

    # This is picking a forum
    for i in tqdm(range(40, max_interleaved_index + 1)):
      this_forum_annotators = next(ann_gen)
      for example in Example.objects.all().filter(interleaved_index=i):
        num_rebuttal_sentences = len(
                Sentence.objects.all().filter(comment_id=example.rebuttal_id))
        for ann in this_forum_annotators:
          assignment = AnnotatorAssignment(
                  example=example, initials=ann,
                  num_rebuttal_sentences=num_rebuttal_sentences,
                  num_completed_sentences=0, is_review_complete=False,
                  is_review_valid=False)
          assignment.save()
