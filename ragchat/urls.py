from django.urls import path
from .views import *

urlpatterns = [
    path("upload_file", upload_file, name="upload_file"),
    path("list_uploaded_files", list_uploaded_files, name="list_uploaded_files"),
    path("retrieve_content", retrieve_content, name="retrieve_content")
]
