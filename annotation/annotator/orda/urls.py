from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('assignments/<str:initials>/', views.assignments, name='assignments'),
    path('rebuttal/<str:initials>/<str:review_id>/<int:index>/',
         views.annotate_rebuttal,
         name='ann_rebuttal'),
    path('review/<str:initials>/<str:review_id>/',
         views.annotate_review,
         name='ann_review'),
    path('agreement/', views.agreement, name='agreement'),
    path('review_submitted/', views.review_submitted, name='review_submitted'),
    path('rebuttal_submitted/', views.rebuttal_submitted, name='rebuttal_submitted'),
]
