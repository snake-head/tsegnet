from django.urls import path
from . import views

urlpatterns = [
    path('toothlist', views.toothlist),
    path('upper/<str:filename>', views.import_upper_file),
    path('lower/<str:filename>', views.import_lower_file),
    path('upperResult/<str:filename>/<str:tooth>', views.display_upper_result)
]
