from django.urls import path
from .views import Du_doan_chu_ky, index, xuly_anh,Du_doan_chu_ky_nhan_dien

urlpatterns = [
    path('', index),
    path('test', Du_doan_chu_ky, name='Du_doan_chu_ky'),
    path('loaddata', xuly_anh, name='xuly_anh'),
    path('testdata', Du_doan_chu_ky_nhan_dien, name='Du_doan_chu_ky_nhan_dien'),
]
