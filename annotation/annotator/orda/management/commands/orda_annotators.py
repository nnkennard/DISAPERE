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
    parser.add_argument('annotators_file', type=str)

  def handle(self, *args, **options):

    print("Currently, the database contains the following annotators:")
    annotator_names = [x.name for x in Annotator.objects.all()]
    print(", ".join(annotator_names))

    result = input(
        ("Are you sure you want to delete all existing annotators and "
         "assignments and repopulate the database? Y/N >"))
    if not result == "Y":
      return

    annotators_file = options["annotators_file"]
    Annotator.objects.all().delete()
    with open(annotators_file, 'r') as f:
      obj = yaml.safe_load(f)
      for annotator in obj["annotators"]:
        ann = Annotator(name=annotator["name"],
                        initials=annotator["initials"],
                        is_staff=False)
        ann.save()
      for annotator in obj["staff"]:
        ann = Annotator(name=annotator["name"],
                        initials=annotator["initials"],
                        is_staff=True)
        ann.save()
