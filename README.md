# 텍스처 분류 시스템

- test이미지와 train이미지의 사이즈가 둘 다 150*150라서 이미지를 crop하지 않고 바로 학습해도 되지만 그렇게 되면 parameter의 개수가 너무 커지기 때문에 다음 두가지 방법으로 모델을 평가
1. 첫번째 방법으로는 train이미지와 test이미지 모두 32*32로 resize한 후 학습 (train 9248개 test 1946개)
2. 두번째 방법으로는 train이미지는 150*150크기의 이미지를 100으로 resize해준 후 하나의 이미지당 랜덤으로 5개의 32*32이미지로 crop해주고 test이미지는 32*32사이즈로 resize해준 후 학습 (train 46240개 test 1946개)
- 학습 데이터의 양이 주는 변화를 보기 위해서 두가지 방법을 시도해보았습니다.
- model1과 model2에서는 (1) GLCM만 (2) GLCM+Law으로 실습해 보았습니다.

### Texture 특징으로 Bayesian classifier
- 우선 이미지에서 GLCM feature를 총 2차원으로(dissimilarity, correlation) 추출했고 label은 0, 1, 2, 3으로 지정
- 추출한 모든 특징들이 독립적이지 않다고 가정하고 분류기 생성 (아래 공식 사용)
![image](https://user-images.githubusercontent.com/56287836/122564050-e6690b80-d07f-11eb-8311-607d2f45e79a.png)


train data에서 평균 벡터, 공분산 행렬(특징 벡터의 개수인 6*6), 사전확률을 계산해서 test data가 들어왔을 때 결정 계수 값을 계산해서 비교 후 제일 큰 값으로 예측하게 됩니다.
Likelihood함수는 위의 세 개의 값을 받아서 위 공식의 계산 값을 반환하게 됩니다.
1. 첫번째 방법에서의 matrix는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564122-f84aae80-d07f-11eb-8825-33f1952f4ab9.png)

정확도는 54%로 나타나고 mountain은 제대로 분류가 되는 것으로 보이고 sea은 오분류가 가장 많게 나타났습니다.

2. 두번째 방법에서의 matrix는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564136-fc76cc00-d07f-11eb-89f6-e5840130dcb3.png)

정확도는 44%로 나타나고 forest와 바다 이외의 class들은 오분류의 확률이 더 높게 나타났습니다.

#### 그래프가 더 좋게 나타난 첫번째 방법 + Law texture feature(2 ➔ 11, 9개 추가)
![image](https://user-images.githubusercontent.com/56287836/122564154-000a5300-d080-11eb-93c2-e8de6a9096da.png)

정확도는 68%로 가장 높게 나타났습니다. Sea이외의 class들은 거의 정확하게 분류했다고 나타납니다.

> 위의 방법들을 비교했을 때 이미지를 같은 사이즈로 resize하고 학습하는 것이 더 높은 정확도를 나타냈고 GLCM texture feature만 쓰는 것 보다 law feature값들을 같이 학습하는 것이 더 높은 정확도를 나타냈습니다. 마지막 방법에서는 sea class이외의 class에서는 정말 높은 정확도를 보여주고 있고 특정 클래스의 오분류가 높아 전체적인 정확도는 68퍼로 그리 높지 않았습니다.

### Texture 특징으로 MLP classifier
- 이미지에서 GLCM feature(dissimilarity, correlation)를 추출해서 train data와 test data를 만들고 신경망을 구축합니다.
Dataset 클래스에는 초기화 함수, Dataset안에있는 데이터의 개수를 반환(라벨의 개수와 동일)하는 len함수, 입력 idx의 feature와 label을 묶어서 반환해주는 getitem함수가 있습니다. 
- 세개의 fully connected layer을 사용했고 활성화 함수로는 relu를 사용했습니다.
- 앞에서 만든 Dataset 클래스를 이용해서 X와 Y를 입력으로 Train/Test data set을 만들고 batch size만큼 랜덤으로 뽑아서 반환해주는 DataLoader를 각각 만들어서 학습할 준비를 마쳤습니다.
- Input, hidden, out 차원을 설정하고 epoch만큼 반복해 학습을 시작했고 맞으면 true, 틀리면 false를 저장해논 evaluation리스트를 통해 accuracy를 계산했습니다.

![image](https://user-images.githubusercontent.com/56287836/122564204-0ac4e800-d080-11eb-8e99-97b1276a08ef.png)

설정해준 epoch(500)와 learning rate(0.01), batch size(10)로 train data와 test data의 accuracy와 loss값을 시각화 했을 때 아래와 같은 그래프를 나타냈습니다.
1. 첫번째 방법에서의 loss/accuracy 그래프는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564218-0dbfd880-d080-11eb-862b-6f5ca4fdf238.png)
![image](https://user-images.githubusercontent.com/56287836/122564228-10223280-d080-11eb-921a-7cc2cdf411d7.png)

Train과 test 모두 점점 loss도 줄고 accuracy도 증가하지만 변화가 너무 미미하고 정확성이 60%를 넘지 못한 것을 볼 수 있습니다.
2. 두번째 방법에서의(learning rate=0.05, hidden dim=11조정) loss/accuracy 그래프는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564247-13b5b980-d080-11eb-9c01-506d2e6c464f.png)
![image](https://user-images.githubusercontent.com/56287836/122564252-157f7d00-d080-11eb-958b-0f888c46edd1.png)

학습이 전혀 제대로 되지 않은 것을 보여주는 그래프입니다.
#### 그래프가 더 좋게 나타난 첫번째 방법 + Law texture feature(2 ➔ 11, 9개 추가)

![image](https://user-images.githubusercontent.com/56287836/122564263-187a6d80-d080-11eb-9aec-c6aeff78339f.png)
![image](https://user-images.githubusercontent.com/56287836/122564272-1c0df480-d080-11eb-8d61-1a3d61952a85.png)
> 세가지 방법을 시도해봤을 때 test data의 loss/accuracy모두 오차 범위가 너무 컸고 높은 정확도를 보여주지 못했습니다.

### RGB pixel값을 입력으로 MLP classifier
- 0~255의 값이 들어가지만 신경망은 작은 값을 더 잘 처리해주기 때문에 -1~1사이의 값으로 만들어줍니다.
- MLP신경망에서 fully connected layer을 5개 사용하고 overfitting을 막기 위해 dropout을 넣어줍니다.
![image](https://user-images.githubusercontent.com/56287836/122564328-2d570100-d080-11eb-8fc8-d4783e9d72c5.png)

설정해준 epoch(100)와 learning rate(0.001), batch size(10)로 train data와 test data의 accuracy와 loss값을 시각화 했을 때 아래와 같은 그래프를 나타냈습니다.
1. 첫번째 방법에서의 loss/accuracy 그래프는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564356-3516a580-d080-11eb-90c3-eda8d781398a.png)
![image](https://user-images.githubusercontent.com/56287836/122564364-38119600-d080-11eb-8928-18d4be290927.png)

epoch를 100으로 설정했지만 epoch 30 이후부터의 loss/accuracy는 많은 변화가 없었습니다. loss는 0.4보다 낮게 나타났고 정확도는 95%정도로 높게 나타났습니다.
2. 두번째 방법(epoch=50)에서의 loss/accuracy 그래프는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564372-3a73f000-d080-11eb-8df3-cafb6ea6fc62.png)
![image](https://user-images.githubusercontent.com/56287836/122564381-3c3db380-d080-11eb-8221-823f6fa5959a.png)

Loss는 점점 감소하고 accuracy는 점점 증가하는 그래프를 보여주지만 정확도는 최대 73%까지만 올라갔습니다. 
> 두가지 방법을 비교했을 때 이미지를 둘 다 32*32로 잘라서 학습하는 것이 더 좋은 성능을 보여줬습니다.

### RGB pixel값을 입력으로 CNN classifier
- CNN의 이미지를 입력할 때는 입력 이미지의 차원의 순서가 batch size , 채널 개수, 이미지의 길이, 이미지의 너비 순으로 입력해야 하기 때문에 swapaxes함수를 사용해 차원을 조절 해주었습니다.
- CNN은 convolution layer을 conv2d을 사용했고 pooling을 사용해 파라미터의 개수를 줄이고 넓게 보도록 해주었습니다.
![image](https://user-images.githubusercontent.com/56287836/122564414-4790df00-d080-11eb-9970-4e9364129c54.png)

설정해준 epoch(100)와 learning rate(0.001), batch size(10)로 train data와 test data의 accuracy와 loss값을 시각화 했을 때 아래와 같은 그래프를 나타냈습니다.
1. 첫번째 방법에서의 loss/accuracy 그래프는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564424-4a8bcf80-d080-11eb-8be0-a396d5df3d7f.png)
![image](https://user-images.githubusercontent.com/56287836/122564429-4c559300-d080-11eb-80e0-58d202ce9087.png)

Train loss는 점점 줄어들지만 test loss는 줄어들다가 epoch20정도부터 다시 높아지기 시작합니다. accuracy는 train은 점점 높아지고 test는 마찬가지로 20정도부터 다시 점점 감소하기 시작합니다. 그래프를 보니 overfitting이 일어났다고 볼 수 있습니다.
2. 두번째 방법에서의 loss/accuracy 그래프는 다음과 같이 나타났습니다.
![image](https://user-images.githubusercontent.com/56287836/122564441-4f508380-d080-11eb-92e2-58698ae5ec8f.png)
![image](https://user-images.githubusercontent.com/56287836/122564447-5081b080-d080-11eb-9800-7fbe39a83053.png)

첫번째 방법에서 overfitting을 막기 위해서 epoch를 50으로 바꾸고 batch size를 20으로 바꾼 뒤 한 번 더 학습해 보았습니다.
![image](https://user-images.githubusercontent.com/56287836/122564451-524b7400-d080-11eb-9710-24cdfb67a194.png)
![image](https://user-images.githubusercontent.com/56287836/122564457-537ca100-d080-11eb-9fd6-b75dbd501cd3.png)

Train data의 accuracy는 학습량이 부족해 더 높아지지는 못했지만 test data에서 Overfitting이 없어진 것을 알 수 있었습니다. 

> 총 4가지 분류기를 모두 학습해본 결과 모두 같은 크기로 resize한 것이 정확도와 loss면에서 더 좋은 성능을 보여줬습니다.

## 비교분석
1. 우선 특징 추출을 사용한 model1과 model2는 두 분류기 모두 정확도가 50~70%로 나타나서 그다지 좋은 성능을 보여주지 않았습니다. Model1, 베이지안 분류기는 GLCM과 Law texture feature을 동시에 사용하는 것이 성능이 좋게 나왔습니다. 반면에 model2, MLP분류기에서는 law feature을 추가하지 않을 때 많은 차이는 없지만 비교적 더 좋은 성능을 보여주었습니다. 하지만 오차 범위는 더 적어지는 것을 볼 수 있었습니다. 
2. MLP를 사용한 model2와 model3에서 model2는 그다지 좋은 성능을 보여주지 않았지만 model3에서는 loss값은 0.2, 정확도는 95%로 높은 성능을 보여줬습니다. 많은 학습 데이터가 있을 때 MLP는 RGB pixel값을 받을 때 더 좋은 성능을 보여준다는 것을 알 수 있었습니다.
3. 마지막으로 CNN 분류기는 그래프를 보았을 때 다른 분류기와는 다르게 완만한 곡선을 이루고 있습니다. 굉장히 좋은 성능을 보여주고 있지만 epoch가 어느 시점을 지났을 때 overfitting이 나타난 것을 볼 수 있습니다. 그래서 epoch를 20으로 바꾸고 배치 사이즈를 20으로 바꿔서 학습한 결과 기존 모델보다 성능은 비슷하지만 overfitting으 나타나지 않았습니다.

### Datasets
https://www.kaggle.com/puneet6060/intel-image-classification
