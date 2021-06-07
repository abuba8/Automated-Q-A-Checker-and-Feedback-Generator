from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

# Create your models here.

class Test(models.Model):
    user_id = models.CharField(max_length = 50)
    test_id = models.AutoField(auto_created=True, primary_key=True)
    test_name = models.CharField(max_length = 100)
    test_no = models.IntegerField()
    test_file = models.FileField(upload_to = 'uploads/')
    grades_file = models.FileField(upload_to = 'uploads/')

    def __str__(self):
        return self.test_name


class Student(models.Model):
    roll_no = models.CharField(max_length=10)
    total_qs = models.IntegerField()
    Obtained_marks = models.IntegerField()
    Percentage = models.FloatField()
    feedback = models.CharField(max_length = 200)
    improvement_feedback = models.CharField(max_length = 200)

    def __str__(self):
        return self.roll_no
    



    