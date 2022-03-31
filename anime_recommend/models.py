from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.
class Anime(models.Model):
    Anime_PlanetID = models.IntegerField(default=0)
    Name = models.CharField(max_length=2000)
    Alternative_Name= models.CharField(max_length=2000)
    Rating_score = models.FloatField(default=0)
    Number_votes = models.IntegerField(default=0)
    Tags = models.CharField(max_length=2000)
    Content_Warning = models.CharField(max_length=2000)
    Type = models.CharField(max_length=2000)
    Episodes = models.IntegerField(default=0)
    Finished = models.CharField(max_length=10)
    Duration = models.IntegerField(default=0)
    StartYear = models.IntegerField(default=0)
    EndYear = models.IntegerField(default=0)
    Season = models.CharField(max_length=2000)
    Studios = models.CharField(max_length=2000)
    Synopsis = models.CharField(max_length=2000)
    Url = models.CharField(max_length=5000)
    picture_url = models.CharField(max_length=5000)
    picture = models.FileField()

    def __str__(self):
        return str(self.Name)


class Myrating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    anime = models.ForeignKey(Anime, on_delete=models.CASCADE)
    rating = models.IntegerField(default=0)


class MyList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    anime = models.ForeignKey(Anime, on_delete=models.CASCADE)
    watch = models.BooleanField(default=False)
