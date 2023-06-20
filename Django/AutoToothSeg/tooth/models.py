from django.db import models
import os
from django.conf import settings


# def upper_jaw_path():
#     return os.path.join(settings.LOCAL_FILE_DIR, 'upper')
#
#
# def lower_jaw_path():
#     return os.path.join(settings.LOCAL_FILE_DIR, 'lower')
#
#
# # Create your models here.
# class UpperJaw(models.Model):
#     file = models.FilePathField('文件路径', path=upper_jaw_path(), match=r'\.stl$')
#     created_time = models.DateTimeField('创建时间', auto_now_add=True)
#     updated_time = models.DateTimeField('更新时间', auto_now=True)
#
#     def __str__(self):
#         return 'filepath %s' % self.file
#
#
# class LowerJaw(models.Model):
#     file = models.FilePathField('文件路径', path=lower_jaw_path(), match=r'\.stl$')
#     created_time = models.DateTimeField('创建时间', auto_now_add=True)
#     updated_time = models.DateTimeField('更新时间', auto_now=True)
#
#     def __str__(self):
#         return 'filepath %s' % self.file
