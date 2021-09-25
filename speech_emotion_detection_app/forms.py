from django import forms

from .models import Audio

class AudioForm(forms.ModelForm):
    class Meta:
        model = Audio
        fields = '__all__'
    
    """def save(self, commit=True):
        temp = super(AudioForm, self).save(commit=False)
        if commit:
            temp.save()
        return temp"""