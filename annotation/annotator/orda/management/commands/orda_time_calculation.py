from django.core.management.base import BaseCommand
from orda.models import *

import datetime


def calc_elapsed_time(a):
  start_time = datetime.datetime.fromtimestamp(int(a.start_time / 1000))
  end_timestamp = a.time_to_annotate + a.start_time / 1000
  end_time = datetime.datetime.fromtimestamp(end_timestamp / 1000)
  return end_time - start_time


class Command(BaseCommand):
  args = '<foo bar ...>'
  help = 'Given a json file, update database for alignments'

  def handle(self, *args, **options):
    for a in ReviewAnnotation.objects.all():
      print("\t".join([
          "Review",
          a.initials,
          a.review_id,
          str(datetime.datetime.fromtimestamp(int(a.start_time / 1000))),
          str(calc_elapsed_time(a)),
      ]))
    for a in RebuttalSentenceAnnotation.objects.all():
      print("\t".join([
          "Rebuttal",
          a.initials,
          a.review_id,
          str(datetime.datetime.fromtimestamp(int(a.start_time / 1000))),
          str(calc_elapsed_time(a)),
      ]))
