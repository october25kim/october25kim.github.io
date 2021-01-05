---
title: Towards open set deep networks
date: 2020-10-11 11:30:00 +0800
categories:
- paper
- openset
tags:
- Open Set Recognition
---


## 논문 선정 이유
강필성 교수님의 비즈니스 어낼리틱스 수업의 첫번째 논문 구현 주제는 **차원 축소**다. 차원 축소와 관련하여 아래의 논문을 선정하였다.<br/>

<div align="center">
<img src="/assets/figures/openmax/openmax0.png" 
title="Towards open set deep networks" 
width="800"/>
</div>  

 >Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1563-1572).

위 논문은 축소된 차원에서 각 클래스별 중심 벡터를 계산한 후, 클래스별 중심 벡터와 떨어진 거리를 활용하여 학습데이터에 없는 클래스의 데이터를 거부하는 방법론을 다룬다.
기존에 연구 중이던 주제이기도 하였기에 이를 직접 구현해보기로 했다. 

### 논문의 특징
>**1) Open Set Recognition**<br>
기존의 딥러닝 기반 Classifier는 학습 단계에서 모든 예측 가능한 클래스가 학습된다는 닫힌 세계 가정을 따른다. 
이로 인해 학습 데이터에 없는 클래스의 데이터에 대해 예측할 때에도 높은 확률로 학습된 클래스 중 하나로 분류하게 된다. 
하지만 실제 세계에서는 수집되지 않은 클래스의 데이터가 대다수이며 단기간에 모든 가능한 클래스의 데이터를 학습시킨다는 것은 불가능에 가깝다. 
따라서 학습되지 않은 클래스의 알람에 대하여 학습된 클래스의 알람으로 오분류 하지 않고,
모르는 클래스로 감지할 수 있는 알고리즘을 Open Set Recognition이라고 한다. (Scheirer et al., 2012).<br><br>
**2) 축소된 차원(Feature Space)에서 Open Set Risk 측정**<br>
해당 논문의 Open Set Recognition 알고리즘은 모델의 학습이 완료된 후에 후처리로 SoftMax 확률값을 새롭게 정의하여(OpenMax) 모르는 클래스의 데이터일
확률값을 도출한다. 이때 모르는 클래스의 데이터일 확률은 네트워크의 마지막에서 두번째 레이어의 Feature Space에서 계산된 Activation Vector를 활용한다.
<br>


## 구현 대상 논문의 방법론 요약

### 학습 직후 후처리 : Extreme Value theorem 기반 클래스별 Outlier Distribution을 도출
<div align="center">
<img src="/assets/figures/openmax/openmax1.png" 
title="EVT Meta-Recognition Calibration for Open Set Deep Networks" 
width="500"/>
</div>
논문에서는 모델의 학습이 완료된 후 Softmax layer에 입력되기 전 마지막에서 두번째 레이어의 Feature Space의 Activation Vector를 활용하여 각 클래스별 Outlier Distribution을 도출한다.  
알고리즘을 순차적으로 따라가면 먼저 학습데이터를 이용하여 Classifier를 학습하고, 학습된 Classifier가 정분류한 학습 데이터만을 추출한 뒤,
각 클래스별로 분리한다. 다음으로 각 클래스별로 Classifier의 마지막에서 두번째 레이어의 Feature Space의 Activation Vector를 수집한 뒤
클래스별 평균 Activation Vector를 계산한 후, 각 정분류 관측치의 Activation Vector가 해당 클래스의 평균 Activation Vector와 떨어진 거리를 계산한다. 
이를 오름차 순으로 정렬하여 각 클래스별 평균 Activation Vector와 떨어진 거리가 먼 순서대로 \\(\eta\\)개의 거리를 샘플로 사용하여 각 클래스별 극단분포를 도출한다. 
논문에서는 \\(\eta\\)개의 먼 거리 샘플로 Weibull distribution에 피팅하여 이를 극단분포로 활용한다. 이렇게 극단치의 샘플들을 이용하여 극단분포를 피팅하는 이론적 배경은 Extreme Value Theorem에 있다.<br>

<div align="center">
<img src="/assets/figures/openmax/openmax4.png" 
title="Extreme Value Theorem"/>
</div>

### 테스트 단계 : Unknown Class에 대한 SoftMax 확률값 정의 - OpenMax
<div align="center">
<img src="/assets/figures/openmax/openmax2.png" 
title="OpenMax probability estimation with rejection of unknown or uncertain inputs"
width="500"/>
</div>

다음으로는 앞 단계의 각 클래스별 극단분포와 평균 Activation Vector를 이용하여 Open Set Recognition 알고리즘을 수행하는 핵심파트이다. 해당 파트에서는 새로운 테스트 데이터가 입력되었을 때
각 클래스별로 테스트 데이터의 Activation Vector와 해당 클래스의 평균 Activation Vector가 떨어진 거리를 계산하고 계산된 거리로 해당 클래스 극단분포의 CDF(cumulative distribution function)에 입력하여
새로운 테스트 데이터가 각 클래스별 극단분포를 어디쯤 위치하는지를 계산한다. CDF의 값이 낮을수록 각 클래스별 평균 Activation Vector와 떨어진 거리가 멀기 때문에 Activation Vector의 각 원소값(각 클래스에 매칭되는)을
CDF 결과값과 곱하여 갱신해주게 되면 해당 클래스에 속하지 않는 데이터의 경우 Activation Vector의 해당 클래스 원소값이 작아지게 되어 SoftMax 확률값 또한 줄어들게 된다. 다음으로 각 클래스별 Activation Vector의 원소값이 줄어든 부분을
모두 모아 새로운 Unknown class의 Activation Vector 원소값을 정의한다. 이로 인해 Unknown Class에 대한 Softmax 확률값을 계산할 수 있고 모델은 수정된 SoftMax(OpenMax)를 통해 새로운 테스트 데이터에 대한 예측 레이블을 결정한다.
또한, Unknown Class Detection 성능을 보완하기 위해 threshold 값을 정하여 학습 데이터의 모든 클래스의 Softmax 확률값이 모두 threshold보다 작을 경우에도 Unknown Class로 예측하게 구성한다.

## 코드 구현
### Classifier 구축
논문의 내용을 구현하기 위해 우선 논문과 동일하게 Image Classification Task를 수행하는 모델을 준비했다.
논문에서는 pre-trained AlexNet을 Base Model로 사용하였지만 해당 논문은 모델 학습 후 후처리로 적용되는 알고리즘이기 때문에 어떠한 Classifier를 사용해도 무방했다.
따라서 이번 과제에서는 3-Dense Block의 DenseNet과 Smaller VGGNet, ResNet을 각각 구현하여 알고리즘의 Base Model로 사용했다.
```python
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# DenseNet
def ConvBlock(x, filters):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters * 4, use_bias=False,
               kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, use_bias=False,
               kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    return x

def TransitionBlock(x, filters, compression=1):
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=int(filters * compression), use_bias=False,
               kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x

def DenseBlock(x, layers, growth_rate):
    concat_feature = x
    for l in range(layers):
        x = ConvBlock(concat_feature, growth_rate)
        concat_feature = Concatenate(axis=-1)([concat_feature, x])
    return concat_feature

def densenet_model(x_shape, y_shape, use_bias=False, print_summary=False):
    _in = Input(shape=x_shape)
    x = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(_in)
    x = DenseBlock(x, 16, 12)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 16, 12)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 16, 12)
    x = GlobalAveragePooling2D()(x)
    _out = Dense(units=y_shape, use_bias=False, activation=None)(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name='DenseNet')
    if print_summary:
        model.summary()
    return model

# Smaller VGGNet
def vggnet_model(x_shape, y_shape, use_bias=False, print_summary=False):
    _in = Input(shape=x_shape)
    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(_in)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # (CONV => RELU) * 2 => POOL
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # (CONV => RELU) * 2 => POOL
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # FC => RELU
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    _out = Dense(units=y_shape, use_bias=False, activation=None)(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name="SmallerVGGNet")
    if print_summary:
        model.summary()
    return model

# ResNet
def ConvBlock1(x):
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def ConvBlock2(x, num_blocks, filter_1, filter_2, first_strides):
    shortcut = x
    for i in range(num_blocks):
        if (i == 0):
            x = Conv2D(filter_1, (1, 1), strides=first_strides, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_1, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_2, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(filter_2, (1, 1), strides=first_strides, padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            shortcut = x
        else:
            x = Conv2D(filter_1, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_1, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_2, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            shortcut = x
    return x

def resnet_model(x_shape, y_shape, use_bias=False, print_summary=True):
    _in = Input(shape=x_shape)
    x = ConvBlock1(_in)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = ConvBlock2(x, num_blocks = 3, filter_1 = 32, filter_2 = 128, first_strides = (1, 1))
    x = ConvBlock2(x, num_blocks = 4, filter_1 = 64, filter_2 = 256, first_strides = (2, 2))
    x = ConvBlock2(x, num_blocks = 6, filter_1 = 128, filter_2 = 512, first_strides = (2, 2))
    x = ConvBlock2(x, num_blocks = 3, filter_1 = 256, filter_2 = 1024, first_strides = (2, 2))
    x = GlobalAveragePooling2D()(x)
    _out = Dense(units=y_shape, use_bias=False, activation=None)(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name='ResNet50')
    if print_summary:
        model.summary()
    return model
```
### 데이터 불러오기 및 전처리

다음으로 실험 및 평가를 위해 사용할 데이터로 CIFAR-10 데이터를 활용했다. 다음 코드는 CIFAR-10 데이터를 불러와서 CNN Classifier에 입력될 수 있도록 전처리를 진행하는 코드이다.

```python
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
from sklearn.preprocessing import OneHotEncoder

def adjust_images(raw_images):
    adj_images = tf.image.per_image_standardization(raw_images).numpy()
    adj_images = np.array([(x-x.min())/(x.max()-x.min()) for x in adj_images], dtype=np.float32)

    return adj_images

total_classes = list(range(10))
target_classes = total_classes
m = len(target_classes)

(train_images, train_y), (test_images, test_y) = load_data()
train_x, test_x = train_images.astype(np.float32) / 255, test_images.astype(np.float32) / 255
train_x, test_x = adjust_images(train_x), adjust_images(test_x)
train_y, test_y = train_y.flatten(), test_y.flatten()

enc = OneHotEncoder(sparse=False, categories='auto')
train_y_enc = enc.fit_transform(train_y.reshape(-1, 1)).astype(np.float32)

train_data = tf.data.Dataset.from_tensor_slices(
    (train_x, train_y_enc)).shuffle(BUFFER_SIZE, SEED, True).batch(BATCH_SIZE)
```
### Classifier 모델 학습
이후 학습데이터를 통해 모델의 학습을 진행한다. 모델의 Optimizer는 Adam Optimizer를 사용하였고 Learning Rate은 0.01로 시작하여 50epoch 이후 0.001, 100epoch 이후 0.0001로 설정하여 점점 줄어들게 하였다. 이후
총 150epoch의 학습을 진행하여 학습을 완료시켰다.

```python
from tensorflow.keras.callbacks import EarlyStopping
from time import time
from tqdm import tqdm

def network_train_step(x, y):
    with tf.GradientTape() as network_tape:
        y_pred = CNN(x, training=True)

        network_loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
        network_acc = tf.keras.metrics.categorical_accuracy(y, y_pred)

    network_grad = network_tape.gradient(network_loss, CNN.trainable_variables)
    network_opt.apply_gradients(zip(network_grad, CNN.trainable_variables))

    return tf.reduce_mean(network_loss), tf.reduce_mean(network_acc)

def train(dataset, epochs):
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        if epoch == 50:
            network_opt.__setattr__('learning_rate', 1E-3)
        elif epoch == 100:
            network_opt.__setattr__('learning_rate', 1E-4)
            
        avg_loss = []
        for batch in dataset:
            losses = network_train_step(batch[0], batch[1])
            avg_loss.append(losses)

        pbar.set_description('Categorical CE Loss: {:.4f} | Accuracy: {:.4f} '.format(
            *np.array(losses)))

# CNN model
if model_name == 'DenseNet':
    CNN = densenet_model(train_x.shape[1:], train_y_enc.shape[1], False, False)
elif model_name == 'resnet':
    CNN = resnet_model(train_x.shape[1:], train_y_enc.shape[1], False, False)
elif model_name == 'SmallerVGGNet':
    CNN = vggnet_model(train_x.shape[1:], train_y_enc.shape[1], False, False)
else :
    print('Model is not defined')
train(train_data, 150)
```
### OpenMax 구축
다음으로 libmr 패키지를 이용하여 Weibull Distribution fitting을 하였고 앞서 요약한 논문의 방법론 중 테스트 단계에 해당하는 OpenMax 알고리즘를 순서에 따라 구현하였다. 
```python
def get_model_outputs(dataset, prob=False):
    pred_scores = []
    for x in dataset:
        model_outputs = CNN(x, training=False)
        if prob:
            model_outputs = tf.nn.softmax(model_outputs)
        pred_scores.append(model_outputs.numpy())
    pred_scores = np.concatenate(pred_scores, axis=0)
    return pred_scores

train_data = tf.data.Dataset.from_tensor_slices(train_x).batch(TEST_BATCH_SIZE)
train_pred_scores = get_model_outputs(train_data, False)
train_pred_simple = np.argmax(train_pred_scores, axis=1)
print(accuracy_score(train_y, train_pred_simple))

train_correct_actvec = train_pred_scores[np.where(train_y == train_pred_simple)[0]]
train_correct_labels = train_y[np.where(train_y == train_pred_simple)[0]]

dist_to_means = []
mr_models, class_means = [], []
for c in np.unique(train_y):
    class_act_vec = train_correct_actvec[np.where(train_correct_labels == c)[0], :]
    class_mean = class_act_vec.mean(axis=0)
    dist_to_mean = np.square(class_act_vec - class_mean).sum(axis=1)
    dist_to_mean = np.sort(dist_to_mean).astype(np.float64)
    dist_to_means.append(dist_to_mean)
    mr = libmr.MR()
    mr.fit_high(dist_to_mean[-eta:], eta)
    class_means.append(class_mean)
    mr_models.append(mr)

class_means = np.array(class_means)

def compute_openmax(actvec):
    dist_to_mean = np.square(actvec - class_means).sum(axis=1).astype(np.float64)
    scores = []
    for dist, mr in zip(dist_to_mean, mr_models):
        scores.append(mr.w_score(dist))
    scores = np.array(scores)
    w = 1 - scores
    rev_actvec = np.concatenate([
        w * actvec,
        [((1 - w) * actvec).sum()]])
    return np.exp(rev_actvec) / np.exp(rev_actvec).sum()

def make_prediction(_scores, _T, thresholding=True):
    _scores = np.array([compute_openmax(x) for x in _scores])
    if thresholding:
        uncertain_idx = np.where(np.max(_scores, axis=1) < _T)[0]
        uncertain_vec = np.zeros((len(uncertain_idx), m + 1))
        uncertain_vec[:, -1] = 1
        _scores[uncertain_idx] = uncertain_vec
    _labels = np.argmax(_scores, 1)
    return _labels


```
### 알고리즘 성능 평가
끝으로 학습된 클래스의 데이터와 학습되지 않은 클래스의 데이터를 통해 성능을 평가한다. MNIST 데이터와 Random Noise 데이터를 Unknown Test Data로 활용하였고 
이를 통해 학습된 클래스의 Test Data의 f1_score와 accuracy, 학습되지 않은 Unknown Test Data에 대한 f1_score와 accuracy를 각각 도출하여 해당 알고리즘의 성능을 평가하였다.
```python
thresholding = True

test_data = tf.data.Dataset.from_tensor_slices(test_x).batch(TEST_BATCH_SIZE)
test_pred_scores = get_model_outputs(test_data)
test_pred_labels = make_prediction(test_pred_scores, threshold, thresholding)

## testing on MNIST (Unseen Classes)
data_train, data_test = tf.keras.datasets.mnist.load_data()
(images_train, labels_train) = data_train
(images_test, labels_test) = data_test
mnist_test = adjust_images(np.array(images_test))
test_batcher = tf.data.Dataset.from_tensor_slices(mnist_test).batch(TEST_BATCH_SIZE)
test_scores = get_class_prob(test_batcher)
test_mnist_labels = make_prediction(test_scores, threshold, thresholding)

## testing on random noise (Unseen Classes)

images = np.random.uniform(0, 1, (10000, 32, 32, 3)).astype(np.float32)
test_batcher = tf.data.Dataset.from_tensor_slices(images).batch(TEST_BATCH_SIZE)
test_scores = get_class_prob(test_batcher)
test_noise_labels = make_prediction(test_scores, threshold, thresholding)

test_unseen_labels = np.concatenate([
        test_mnist_labels,
        test_noise_labels])
    
test_pred = np.concatenate([test_pred_labels, test_unseen_labels])
test_true = np.concatenate([test_y.flatten(),
                            np.ones_like(test_unseen_labels)*m])
    
test_macro_f1 = f1_score(test_true, test_pred, average='macro')
#print(f1_score(test_true, test_pred, average=None))

test_seen_acc = accuracy_score(test_y, test_pred_labels)

test_unseen_f1 = np.array([f1_score(np.ones_like(test_unseen_labels), test_unseen_labels == m),
                           f1_score(np.ones_like(test_mnist_labels), test_mnist_labels == m),
                           f1_score(np.ones_like(test_noise_labels), test_noise_labels == m)])
 
print('overall f1: {:.4f}'.format(test_macro_f1))
print('seen acc: {:.4f}'.format(test_seen_acc))
print('unseen f1: {:.4f} / {:.4f} / {:.4f} / {:.4f} / {:.4f}'.format(*test_unseen_f1))
```

## 결과
총 3개의 CNN Classifier 중 ResNet 모델을 load하여 SoftMax만 적용한 모델의 성능과 OpenMax를 적용한 모델을 비교함으로써 논문의 알고리즘의 성능을 검증하였다.
총 30,000 개의 테스트 데이터 중 학습된 클래스 10,000개의 테스트 데이터에 대해서는 SoftMax가 Accuracy 기준 82.7%, OpenMax가 81.3%의 성능을 보였다.
학습된 클래스의 데이터에 대해서는 OpenMax 알고리즘이 Unknown Class에 대한 SoftMax 확률값을 만들어 주면서 기존의 학습된 클래스의 Activation Vector 원소값을 덜어내기 때문에
학습된 클래스로 분류될 확률이 줄어든다. 따라서 학습된 클래스에 대해서는 OpenMax가 SoftMax 대비 다소 낮은 성능을 보이게 된다.
하지만 학습되지 않은 클래스의 테스트 데이터 20,000개에서는 SoftMax가 Accuracy 기준 7.87%, OpenMax가 31.3%로 큰 성능 차이를 보였다. 
(SoftMax의 경우, 모든 클래스의 SoftMax 확률값이 threshold 0.9보다 작으면 Unknown Class로 분류되게끔 구성) 이를 통해 OpenMax 알고리즘이 Open Set Recognition 성능면에서 효과가 있음을 확인할 수 있다.

>**참고문헌**
1. Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1563-1572).<br>
2. Scheirer, W. J., Rocha, A., Micheals, R. J., & Boult, T. E. (2011). Meta-recognition: The theory and practice of recognition score analysis. IEEE transactions on pattern analysis and machine intelligence, 33(8), 1689-1695.** <br>
