from django.shortcuts import render

# Create your views here.
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from .serializers import UploadSerializer
from .tagmodeltestservice import predict

# ViewSets define the view behavior.


class UploadViewSet(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        file_uploaded = request.FILES.get('file_uploaded')
        if file_uploaded: 
            return Response(predict(file_uploaded))
        else:
            raise NotImplementedError
