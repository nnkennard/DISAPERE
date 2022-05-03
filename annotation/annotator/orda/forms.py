from django import forms

class AnnotationForm(forms.Form):
    annotation = forms.CharField(label='JSON annotation', max_length=20000)



