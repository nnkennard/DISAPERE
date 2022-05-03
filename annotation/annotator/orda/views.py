from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .forms import *
from .models import *

from .orda_utils import *
from .agreement_lib import *

import json
# Create your views here.


def index(request):
  template = loader.get_template('orda/index.html')
  context = {"annotators": Annotator.objects.filter(is_staff=False),
  "staff":Annotator.objects.filter(is_staff=True)}
  return HttpResponse(template.render(context, request))


def assignments(request, initials):

  annotator_name = Annotator.objects.get(initials=initials).name
  previously_completed, assignments_to_show = get_this_annotator_assignments(
      initials)
  presentation_examples = [
      make_presentation_example(a) for a in assignments_to_show
  ]
  template = loader.get_template('orda/assignment.html')
  context = {
      "previously_completed": previously_completed,
      "examples": presentation_examples,
      "annotator": {
          "name": annotator_name,
          "initials": initials
      }
  }
  return HttpResponse(template.render(context, request))


def annotate_review(request, review_id, initials):
  example = Example.objects.get(review_id=review_id)
  review_sentences = get_htmlified_sentences(review_id)
  template = loader.get_template('orda/annotate_review.html')
  return HttpResponse(
      template.render(
          {
              "text": {
                  "review_sentences": review_sentences
              },
              "labels": FORMATTED_LABELS,
              "form": AnnotationForm(),
              "menu_map": ALLOWED_MENUS,
              "metadata": {
                  "paper_title": example.title,
                  "initials": initials,
                  "reviewer": example.reviewer,
                  "forum_id": example.forum_id,
                  "review_id": example.review_id
              }
          }, request))


def review_submitted(request):

  form = AnnotationForm(request.POST)
  if form.is_valid():
    annotation = json.loads(form.cleaned_data["annotation"])
    rev_ann_info = annotation["review_annotation"]

    review_annotation = ReviewAnnotation(
        review_id=rev_ann_info["review_id"],
        overall_comment=rev_ann_info["overall_comment"],
        is_valid=rev_ann_info["is_valid"],
        errors=rev_ann_info["errors"],
        initials=rev_ann_info["initials"],
        time_to_annotate=rev_ann_info["time_to_annotate"],
        start_time=rev_ann_info["start_time"],
    )
    review_annotation.save()

    example_id = Example.objects.get(review_id=rev_ann_info["review_id"]).id

    assignment = AnnotatorAssignment.objects.get(
        initials=rev_ann_info["initials"], example_id=example_id)
    assignment.is_review_complete = True
    assignment.is_review_valid = rev_ann_info["is_valid"]
    assignment.save()

    for rev_sent_ann in annotation["review_sentence_annotations"]:
      sentence_annotation = ReviewSentenceAnnotation(
          review_id=rev_sent_ann["review_id"],
          review_sentence_index=rev_sent_ann["review_sentence_index"],
          initials=rev_sent_ann["initials"],
          labels=json.dumps(rev_sent_ann["labels"]))
      sentence_annotation.save()

    template = loader.get_template('orda/review_submitted.html')
    return HttpResponse(
        template.render({"initials": rev_ann_info["initials"]}, request))
  else:
    return HttpResponse(
        "There has been an error, please use the back button in your browser to retrieve the JSON value of the annotation and share it with Neha. Thanks!"
    )


def annotate_rebuttal(request, initials, review_id, index):
  example = Example.objects.get(review_id=review_id)
  context = get_annotation_context(initials, review_id, index)
  context.update({
      "alignment_categories": ALIGNMENT_CATEGORIES,
      "labels": REBUTTAL_RELATIONS,
      "form": AnnotationForm(),
      "metadata": {
          "paper_title": example.title,
          "initials": initials,
          "reviewer": example.reviewer,
          "forum_id": example.forum_id,
          "rebuttal_id": example.rebuttal_id,
          "review_id": example.review_id,
          "rebuttal_index": index
      }
  })
  template = loader.get_template('orda/annotate_rebuttal.html')
  return HttpResponse(template.render(context, request))


def rebuttal_submitted(request):

  form = AnnotationForm(request.POST)
  if form.is_valid():
    reb_ann_info = json.loads(form.cleaned_data["annotation"])
    print(reb_ann_info)

    rebuttal_annotation = RebuttalSentenceAnnotation(
        review_id=reb_ann_info["review_id"],
        rebuttal_id=reb_ann_info["rebuttal_id"],
        rebuttal_sentence_index=reb_ann_info["rebuttal_sentence_index"],
        initials=reb_ann_info["initials"],
        is_valid=reb_ann_info["is_valid"],
        aligned_review_sentences=json.dumps(
            reb_ann_info["aligned_review_sentences"]),
        relation_label=reb_ann_info["relation_label"],
        comment=reb_ann_info["comment"],
        alignment_category=reb_ann_info["alignment_category"],
        errors=json.dumps(reb_ann_info["errors"]),
        time_to_annotate=reb_ann_info["time_to_annotate"],
        start_time=reb_ann_info["start_time"],
    )
    rebuttal_annotation.save()

    example = Example.objects.get(review_id=reb_ann_info["review_id"])

    assignment = AnnotatorAssignment.objects.get(
        initials=reb_ann_info["initials"], example_id=example.id)
    latest_annotations = get_latest_rebuttal_annotation(
        example, reb_ann_info["initials"])
    num_annotated = len([x for x, _ in latest_annotations if x is not None])
    assignment.num_completed_sentences = num_annotated
    assignment.save()

    maybe_next_index = reb_ann_info["rebuttal_sentence_index"] + 1
    if (maybe_next_index == example.num_rebuttal_sentences):
      next_sentence_info = {
          "valid": False,
          "initials": reb_ann_info["initials"]
      }
    else:
      next_sentence_info = {
          "rebuttal_sentence_index": maybe_next_index,
          "initials": reb_ann_info["initials"],
          "review_id": reb_ann_info["review_id"],
          "valid": True
      }
      print(next_sentence_info)

    template = loader.get_template('orda/rebuttal_submitted.html')
    return HttpResponse(
        template.render(
            {
                "initials": reb_ann_info["initials"],
                "next_sentence_info": next_sentence_info,
            }, request))
  else:
    return HttpResponse(
        "There has been an error, please use the back button in your browser to retrieve the JSON value of the annotation and share it with Neha. Thanks!"
    )


def agreement(request):
  if request.user.is_superuser:
    template = loader.get_template('orda/agreement.html')
    context = {"info": agreement_calculation()}
    return HttpResponse(template.render(context, request))
  else:
    return HttpResponse("Please log in to see this page.")
  
