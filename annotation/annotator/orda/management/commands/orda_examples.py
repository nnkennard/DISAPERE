from django.core.management.base import BaseCommand
from orda.models import *

import json
from tqdm import tqdm

DATASETS = ["traindev_train", "traindev_dev", "traindev_test", "truetest"]


def enter_pair(pair, dataset, interleaved_list):
  interleaved_index = interleaved_list.index(pair["forum"])
  example = Example(
      dataset=dataset,
      example_index=pair["index"],
      forum_id=pair["forum"],
      review_id=pair["review_sid"],
      rebuttal_id=pair["rebuttal_sid"],
      title=pair["title"],
      reviewer=pair["review_author"],
      interleaved_index=interleaved_index,
      num_rebuttal_sentences=len(pair["rebuttal_text"]["sentences"]),
  )
  example.save()

  for review_or_rebuttal in "review rebuttal".split():
    text = pair[review_or_rebuttal + "_text"]["text"]
    for i, sentence_info in enumerate(pair[review_or_rebuttal +
                                           "_text"]["sentences"]):
      sentence_text = text[
          sentence_info["start_index"]:sentence_info["end_index"]]
      sentence = Sentence(comment_id=pair[review_or_rebuttal + "_sid"],
                          sentence_index=i,
                          text=sentence_text,
                          suffix=sentence_info["suffix"])
      sentence.save()


class Command(BaseCommand):
  args = '<foo bar ...>'
  help = 'Given a json file, update database for alignments'

  def add_arguments(self, parser):
    parser.add_argument('review_dir', type=str)

  def _load_data(self, input_file):
    with open(input_file, 'r') as f:
      return json.loads(f.read())

  def handle(self, *args, **options):

    result = input(
        "Are you sure you want to delete all reviews and rebuttals and repopulate the database? Y/N >"
    )
    if not result == "Y":
      return

    Example.objects.all().delete()
    Sentence.objects.all().delete()

    with open(options["review_dir"] + "/interleaved.json", 'r') as f:
      interleaved_list = json.load(f)["interleaved_forum_ids"]

    interleaved_list = ["example_forum"] + interleaved_list

    for dataset in DATASETS:
      print("".join([options["review_dir"], "/", dataset, ".json"]))
      input_file = "".join([options["review_dir"], "/", dataset, ".json"])
      json_obj = self._load_data(input_file)
      print(len(json_obj["review_rebuttal_pairs"]))
      for pair in tqdm(json_obj["review_rebuttal_pairs"]):
        enter_pair(pair, dataset, interleaved_list)

    with open(options["review_dir"] + "/mini_example.json", 'r') as f:
      pair = json.load(f)
      enter_pair(pair, "example", interleaved_list)
