import os
import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import joblib

# Bước 1: Thu thập dữ liệu
chu_ky_that = ".\\signatures\\chu_ky_that"
chu_ky_mo_phong = ".\\signatures\\chu_ky_mo_phong"

# Gọi lại tùy chỉnh để lập biểu đồ 
class Bieu_do_LossAccuracy(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])

        plt.figure(figsize=(12, 4))

        # Biểu đồ Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 2), self.losses, label='Độ mất mát đào tạo')
        plt.plot(range(1, epoch + 2), self.val_losses, label='Độ mất mát xác thực')
        plt.title('Độ mất mát qua mỗi chu kỳ huấn luyện')
        plt.xlabel('chu kỳ huấn luyện')
        plt.ylabel('Độ mất mát')
        plt.legend()

        # Biểu đồ Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 2), self.accuracies, label='Độ chính xác đào tạo')
        plt.plot(range(1, epoch + 2), self.val_accuracies, label='Độ chính xác xác thực')
        plt.title('Độ chính xác qua mỗi chu kỳ huấn luyện')
        plt.xlabel('Chu kỳ huấn luyện')
        plt.ylabel('Độ chính xác')
        plt.legend()

        plt.tight_layout()
        plt.show()

def tai_anh(chu_ky_that, chu_ky_mo_phong):
    data = []
    labels = []

    for file in os.listdir(chu_ky_that):
        if file.endswith(".png"):
            img = io.imread(os.path.join(chu_ky_that, file), as_gray=True)
            img = transform.resize(img, (128, 128))
            data.append(img.flatten())
            labels.append(1)

    for file in os.listdir(chu_ky_mo_phong):
        if file.endswith(".png"):
            img = io.imread(os.path.join(chu_ky_mo_phong, file), as_gray=True)
            img = transform.resize(img, (128, 128))
            data.append(img.flatten())
            labels.append(0)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

# Bước 2: Xử lý trước dữ liệu
data, labels = tai_anh(chu_ky_that, chu_ky_mo_phong)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Bước 3: Xây dựng CNN để trích xuất tính năng
cnn_model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Biên dịch và đào tạo CNN
bieu_do = Bieu_do_LossAccuracy()
cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train.reshape((-1, 128, 128, 1)), y_train, batch_size=32, epochs=30, 
              validation_data=(X_test.reshape((-1, 128, 128, 1)), y_test),
              callbacks=[bieu_do])
# Lưu mô hình CNN đã huấn luyện
cnn_model.save("cnn_xgb_model.h5")

# Bước 4: Trích xuất đặc trưng bằng CNN
feature_extractor = Sequential(cnn_model.layers[:-1])
X_train_features = feature_extractor.predict(X_train.reshape((-1, 128, 128, 1)))
X_test_features = feature_extractor.predict(X_test.reshape((-1, 128, 128, 1)))

# Bước 5: Huấn luyện XGBoost bằng các tính năng được trích xuất
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_features.reshape((X_train_features.shape[0], -1)), y_train)

# Lưu mô hình XGBoost đã đào tạo
joblib.dump(xgb_model, "xgb_model.joblib")
