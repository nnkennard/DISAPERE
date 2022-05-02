from django.core.management.base import BaseCommand
from orda.models import *

import json
import random
from tqdm import tqdm
import yaml

random.seed(43)


class Command(BaseCommand):
  args = '<foo bar ...>'
  help = 'Given a json file, update database for alignments'

  def add_arguments(self, parser):
    parser.add_argument('initials', type=str)
    parser.add_argument('review_id', type=str)

  def handle(self, *args, **options):
    e = Example.objects.get(review_id=options["review_id"])
    relevant_assignments = AnnotatorAssignment.objects.filter(example=e)
    for a in relevant_assignments:
        if a.initials == options["initials"]:
            print(" ".join([options["review_id"], "already assigned to", options["initials"] + "."]))
            return
    k = relevant_assignments[0]
    new_a = AnnotatorAssignment(example=e,
            initials=options["initials"],
            is_review_complete=False,
            is_review_valid=False,
            num_rebuttal_sentences=k.num_rebuttal_sentences,
            num_completed_sentences=0)
    new_a.save()

    print(" ".join([options["review_id"], "assigned to", options["initials"] + "."]))

