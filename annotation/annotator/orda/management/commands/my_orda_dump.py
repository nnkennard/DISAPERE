from django.core.management.base import BaseCommand
from orda.models import *

import json
import subprocess

TABLE_MAP = {
        "text": "example sentence".split(),
        "annotations": ("reviewannotation reviewsentenceannotation "
                        "rebuttalsentenceannotation").split(),
        "assignments": "example annotatorassignment".split(),
        }

class Command(BaseCommand):
  args = '<foo bar ...>'
  help = 'Given a json file, update database for alignments'

  def add_arguments(self, parser):
    parser.add_argument('data_type', type=str)
    parser.add_argument('output_file', type=str)

  def handle(self, *args, **options):
    data_map = {}
    assert options["data_type"] in TABLE_MAP
    tables = TABLE_MAP[options["data_type"]]
    for table in tables:
        table_json = subprocess.run(
            ("python manage.py dumpdata orda." + table).split(),
            capture_output=True).stdout
        data_map[table] = json.loads(table_json)

    with open(options["output_file"], 'w') as f:
        json.dump(data_map, f)
