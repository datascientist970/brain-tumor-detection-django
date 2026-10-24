from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    result = models.CharField(max_length=10)  # To store 'yes' or 'no'
    area_mm2 = models.FloatField(null=True, blank=True)  # To store the area in mm²

    def __str__(self):
        return f"{self.image.name} - {self.result}"
