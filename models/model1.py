from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as sg
import numpy as np
import itertools
import cv2
import os

# === laws texture 계산 함수 ===
def laws_texture(gray_image):
    (rows, cols) = gray_image.shape[:2]
    smooth_kernel = (1/25)*np.ones((5,5))
    gray_smooth = sg.convolve(gray_image, smooth_kernel,"same")
    gray_processed = np.abs(gray_image - gray_smooth)

    filter_vectors = np.array([[1, 4, 6, 4, 1],
                                [-1, -2, 0, 2, 1],
                                [-1, 0, 2, 0, 1],
                                [1, -4, 6, -4, 1]])
   
    filters = []
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5,1), filter_vectors[j][:].reshape(1,5)))

    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    # === 9+1개 중요한 texture map 계산 ===
    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2)
    texture_maps.append((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2)
    texture_maps.append((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2)
    texture_maps.append((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2)
    texture_maps.append((conv_maps[:, :, 6]+conv_maps[:, :, 9])//2)
    texture_maps.append((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2)
    texture_maps.append(conv_maps[:, :, 10])
    texture_maps.append(conv_maps[:, :, 5])
    texture_maps.append(conv_maps[:, :, 15])
    texture_maps.append(conv_maps[:, :, 0])

    # == Law's texture energy 계산 ===
    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())

    return TEM

# == 이미지 패치에서 특징 추출 ===
train_dir = './texture_data/train'
test_dir = './texture_data/test'
classes = ['buildings', 'forest', 'mountain', 'sea']

X_train = []
Y_train = []

PATCH_SIZE = 30
np.random.seed(1234)
for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(train_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_s = cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR)
        
        image_s_gray = cv2.cvtColor(image_s, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(image_s_gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
        X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0], greycoprops(glcm, 'correlation')[0, 0]]+laws_texture(image_s_gray))
        Y_train.append(idx)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print('train data:  ', X_train.shape)
print('train label: ', Y_train.shape)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_s = cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR)
        image_gray = cv2.cvtColor(image_s, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
        X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0], greycoprops(glcm, 'correlation')[0, 0]]+laws_texture(image_gray))
        Y_test.append(idx)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data:  ', X_test.shape)
print('test label: ', Y_test.shape)

priors = []
covariances = []
means = []

# === Bayesian classifier ===
for i in range(len(classes)):
    X = X_train[Y_train == i]
    priors.append((len(X) / len(X_train)))
    means.append(np.mean(X, axis=0))
    covariances.append(np.cov(np.transpose(X), bias=True))

# === likelihood 계산 함수 ===
def likelihood(x, prior, mean, cov):
    return -0.5 * np.linalg.multi_dot([np.transpose(x-mean), np.linalg.inv(cov), (x-mean)]) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)

Y_pred = []
for i in range(len(X_test)):
    likelihoods = []
    for j in range(len(classes)):
        likelihoods.append(likelihood(X_test[i], priors[j], means[j], covariances[j]))
    Y_pred.append(likelihoods.index(max(likelihoods)))
acc = accuracy_score(Y_test, Y_pred)
print('accuracy: ', acc)

# === confusion matrix 시각화 ===
def plot_confusion_matrix(cm, target_names=None, labels=True):
    accuracy = np.trace(cm) / float(np.sum(cm))

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    thresh = cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(confusion_matrix(Y_test, Y_pred), target_names=classes)