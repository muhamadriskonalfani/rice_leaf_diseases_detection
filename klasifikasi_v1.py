import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split

def process_file(file_path):
    # Misalnya kita akan membuat sebuah pesan sederhana berdasarkan nama file
    file_name = file_path.split('/')[-1]
    
    def load_images_from_folder(folder_path):
        images = []
        labels = []
        class_names = os.listdir(folder_path)
        for class_name in class_names:
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_name)
        return images, labels

    def preprocess_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.equalizeHist(gray_image)
        denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        return denoised_image

    def extract_hog_features(image):
        features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        return features

    # Memuat dataset dari folder lokal
    folder_path = './rice_leaf_diseases'  # Pastikan path ini sesuai dengan lokasi folder Anda
    images, labels = load_images_from_folder(folder_path)

    # Pra-pemrosesan setiap gambar
    preprocessed_images = [preprocess_image(cv2.resize(image, (384, 128))) for image in images]

    # Ekstraksi fitur HOG untuk setiap gambar yang sudah dipra-pemrosesan
    features = [extract_hog_features(image) for image in preprocessed_images]

    # Menyatukan fitur HOG menjadi satu array homogen
    X = np.array(features)
    y = np.array(labels)

    # Membagi dataset menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    # Inisialisasi model SVM
    model = svm.SVC(kernel='linear')

    # Melatih model SVM
    model.fit(X_train, y_train)

    # Fungsi untuk mengklasifikasikan gambar baru
    def classify_new_image(image):
        # Resize dan pra-pemrosesan gambar
        resized_image = cv2.resize(image, (384, 128))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.equalizeHist(gray_image)
        denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

        # Ekstraksi fitur HOG
        hog_features = extract_hog_features(denoised_image)
        hog_features = np.array(hog_features).reshape(1, -1)

        # Prediksi kelas
        prediction = model.predict(hog_features)

        return prediction[0]

    # Contoh penggunaan untuk klasifikasi gambar baru yang diberikan sebagai variabel
    # new_image harus berupa array numpy yang merepresentasikan gambar
    new_image = cv2.imread(file_path)  # Gantilah './DSC_0367.JPG' dengan path gambar yang ingin diklasifikasikan
    prediksi = classify_new_image(new_image)
    # print(f'Prediksi untuk gambar baru: {prediksi}')
    
    custom_message = f'File "{file_name}" telah diproses. {prediksi}'
    return custom_message
