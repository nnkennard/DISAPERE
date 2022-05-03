from django.contrib import admin
from .models import *

for model_name in [Example, Sentence, RebuttalSentenceAnnotation,
        ReviewSentenceAnnotation, ReviewAnnotation, Annotator,
        AnnotatorAssignment]:
    admin.site.register(model_name)
