from django.shortcuts import render
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.models import Sequential
from skimage import io, transform
import cv2
import shutil
import os
from skimage import measure, morphology
# from sklearn.metrics import accuracy_score
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np

def index(request):
    return render(request,'index.html')

def xuly_anh(request):
    constant_parameter_1 = 84
    constant_parameter_2 = 250
    constant_parameter_3 = 100
    constant_parameter_4 = 18

    if request.method == 'POST':
        # Đọc dữ liệu ảnh được tải
        image_file = request.FILES['image']
        image_array = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Xử lý ảnh
        img = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)[1] 
        blobs = img > img.mean()
        blobs_labels = measure.label(blobs, background=1)

        the_biggest_component = 0
        total_area = 0
        counter = 0
        average = 0.0

        for region in regionprops(blobs_labels):
            if region.area > 10:
                total_area += region.area
                counter += 1
            if region.area >= 250:
                if region.area > the_biggest_component:
                    the_biggest_component = region.area

        average = total_area / counter
        print("thành phần lớn nhất: " + str(the_biggest_component))
        print("trung bình: " + str(average))
        a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
        print("hằng số ngoại lệ kích thước nhỏ: " + str(a4_small_size_outliar_constant))
        a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4
        print("hằng số ngoại lệ kích thước lớn: " + str(a4_big_size_outliar_constant))
        pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)

        component_sizes = np.bincount(pre_version.ravel())
        too_small = component_sizes > a4_big_size_outliar_constant
        too_small_mask = too_small[pre_version]
        pre_version[too_small_mask] = 0
        pre_version = pre_version.astype(np.uint8)
        # Lưu ảnh
        plt.imsave('kq.jpg', pre_version)

        # Đọc ảnh trước
        img = cv2.imread('kq.jpg', 0)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # Lưu kết quả
        cv2.imwrite("hinhanh.jpg", img)
        source_directory = ''
        destination_directory = 'app_doan\static\images'
        
        file_name = 'hinhanh.jpg'
        
        # Tạo đường dẫn đầy đủ cho tệp nguồn và đích
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        
        # Sao chép file
        shutil.copy(source_file_path, destination_file_path)
    return render(request, 'upload.html')

def Du_doan_chu_ky(request):
    result = None
    accuracy = None
    if request.method == 'POST':
        image_file = request.FILES['image']
        img = io.imread(image_file, as_gray=True)
        img = transform.resize(img, (128, 128))
        img = img.flatten()
        
        # Tải mô hình CNN
        cnn_model = load_model("app_doan/cnn_xgb_model.h5")

        # Trích xuất các tính năng bằng mô hình CNN đã tải
        feature_extractor = Sequential(cnn_model.layers[:-1])
        img_features = feature_extractor.predict(img.reshape((1, 128, 128, 1)))

        # Tải mô hình XGB
        XGB_model = joblib.load("app_doan/xgb_model.joblib")
        
        # Dự đoán bằng mô hình XGB đã tải
        prediction = XGB_model.predict(img_features.reshape((1, -1)))
        probability = XGB_model.predict_proba(img_features.reshape((1, -1)))
        accuracy = probability[0][1] * 100  # Confidence of the positive class in percentage
        
        if(prediction[0] >= 0.5):
            if(accuracy>50):
                 result = "Chữ ký chính chủ"
            else:
                result= "Chữ ký không chính chủ"
        else:
            accuracy=100-accuracy
            result= "Chữ ký không chính chủ"
            
        # return JsonResponse({'result': result, 'confidence': confidence})    
    return render(request, 'test.html', {'result': result, 'accuracy': accuracy})

def Du_doan_chu_ky_nhan_dien(request):
    result1 = None
    accuracy1 = None
    image_file = './app_doan/static/images/hinhanh.jpg'
    
    if request.method == 'POST':
        img = io.imread(image_file, as_gray=True)
        img = transform.resize(img, (128, 128))
        img = img.flatten()

        # Tải mô hình CNN
        cnn_model = load_model("app_doan/cnn_xgb_model.h5")

        # Trích xuất các tính năng bằng mô hình CNN đã tải
        feature_extractor = Sequential(cnn_model.layers[:-1])
        img_features = feature_extractor.predict(img.reshape((1, 128, 128, 1)))

        # Tải mô hình XGB
        XGB_model = joblib.load("app_doan/xgb_model.joblib")

        # Dự đoán bằng mô hình XGB đã tải
        prediction = XGB_model.predict(img_features.reshape((1, -1)))
        probability = XGB_model.predict_proba(img_features.reshape((1, -1)))
        accuracy1 = probability[0][1] * 100  # Confidence of the positive class in percentage
        
        if(prediction[0] >= 0.5):
            if(accuracy1>50):
                 result1 = "Chữ ký chính chủ"
            else:
                result1= "Chữ ký không chính chủ"
        else:
            accuracy1=100-accuracy1
            result1= "Chữ ký không chính chủ"

    return render(request, 'testdata.html', {'result1': result1, 'accuracy1': accuracy1})

