import cv2
import numpy as np

def detect_signature(image_path, template_path):
    # Đọc ảnh chữ ký và ảnh mẫu
    signature_img = cv2.imread(image_path, 0)
    template_img = cv2.imread(template_path, 0)

    # Sử dụng thuật toán Template Matching để tìm chữ ký trong ảnh
    result = cv2.matchTemplate(signature_img, template_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Vị trí tốt nhất của chữ ký
    top_left = max_loc
    h, w = template_img.shape

    # Vẽ hình chữ nhật xung quanh chữ ký
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(signature_img, top_left, bottom_right, 255, 2)

    # Hiển thị ảnh gốc và ảnh đã nhận dạng
    cv2.imshow('Signature Detection', signature_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Sử dụng ví dụ
detect_signature('thuong.png', 'ts.png')
