---
title: A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting
date: 2020-12-06 11:30:00 +0800
categories:
- paper
- ensemble learning
tags:
- anogan
---

## 논문 선정
강필성 교수님의 비즈니스 어낼리틱스 수업의 네번째 논문 구현 주제는 **Ensemble Learning**이다. Boosting Algorihtm 기반의 방법론을 깊게 공부해보고자 
가장 초기의 Boosting Algorihtm 중 하나인 Adaptive Boosting을 다룬 논문을 선정하였다. Adaptive Boosting의 줄임말인 AdaBoost는 1996년에 Freund와 Schapire이 제안한 알고리즘으로 
2003년에는 괴델상을 수상한 알고리즘이기도 하다.
<br/>

<div align="center">
<img src="/assets/figures/adaboost/adaboost.png" 
title="A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting" 
width="600"/>
</div>  

> Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.


## AdaBoost
"Adaptive Boosting"의 약자 인 AdaBoost는 1996년 Freund와 Schapire가 제안한 최초의 실용적인 부스팅 알고리즘이다. 
AdaBoost의 목표를 간략히 말하면 약한 분류기를 강력한 분류기로 변환하는 것이라 할 수 있다. 즉, AdaBoost의 분류를 위한 최종 방정식을 살펴보면 다음과 같다.

$$F(x)=sign(\sum_{m=1}^{M}\theta_m f_m(x))$$

위의 식에서 $$f_m(x)$$는 총 $$M$$개의 개별 약한 분류기들을 나타내고 개별 가중치 $$\theta_m$$을 반영하여 가중합을 분류를 위한 최종 방정식(최종 강한 분류기)를 구축한다.

## AdaBoost algorithm
AdaBoost 알고리즘의 전체 과정은 다음과 같이 요약할 수 있다.
먼저 n개의 데이터 포인트로 구성된 데이터 세트가 주어지면 각각의 데이터 포인트 $$x_i$$에 대하여 레이블 $$y_i$$를 
-1은 negative class를 나타내고 1은 positive class를 나타내도록 구성할 수 있다.

$$x_i \in \mathbb{R}^d, y_i \in \{-1,1\}$$

다음으로 각 데이터 포인트에 대한 가중치는 다음과 같이 초기화한다.

$$w(x_i,y_i)= \frac{1}{n}, i=1,\ldots,n$$

이후, 총 $$M$$개의 약한 분류기에 대해 아래의 과정을 수행한다.

(1) 각 m번째 시행마다 약한 분류기(ex. Decision Tree)를 데이터 세트로 한번 학습시킨 뒤에 분류 오류를 계산한다.

$$\epsilon_m = E_{w_m}[1_{y \neq f(x)}]$$

(2) m번째 약한 분류기의 개별 가중치 $$\theta_m$$을 다음의 식에 따라 계산한다.

$$\theta_m=\frac{1}{2}ln(\frac{1-\epsilon_m}{\epsilon_m})$$

이때, (2)의 개별 가중치 계산식에 따라 분류 정확도가 50% 이상인 경우 가중치는 양수가 되고 각 개별 분류기가 정확할수록 가중치가 커진다. 
반대로 정확도가 50% 미만인 분류기의 경우 가중치는 음수가 되는데 이는 정확도가 50% 미만인 경우 음의 가중치로서 최종 예측에 반영이 됨을 의미한다.
즉, 50% 정확도를 가진 분류기는 아무런 정보를 추가하지 않으므로 최종 예측에 영향을 주지 않는 반면, 정확도가 40%인 분류기는 음의 가중치로 페널티를 가지면서 최종 예측에 기여하게 된다.

(3) 다음으로 각 데이터별 가중치를 업데이트 한다.

$$w_{m+1}(x_i,y_i)=\frac{w_m(x_i,y_i)exp[-\theta_m y_i f_m(x_i)]}{Z_m}$$

이 때 $$Z_m$$은 모든 데이터별 가중치의 총합이 1이 되도록하는 Normalization Factor이다.

위 식을 살펴보면 분류기가 잘못 분류한 데이터 포인트의 경우, 분자에서 지수항($$exp[-\theta_m y_i f_m(x_i)]$$)이 항상 1보다 크게 된다.

$$\because y_i f_m(x_i)=-1 \, \And \, \theta_m\ge0$$

따라서 잘못 분류한 데이터 포인트는 (3)의 과정을 거치고 나면 더 큰 가중치로 업데이트된다. 이 (1)~(3)의 과정을 $$M$$개의 약한 분류기에 대해 모두 수행한 뒤, 각 분류기의 가중합을 통해 최종 예측을 얻는다.

$$F(x)=sign(\sum_{m=1}^{M}\theta_m f_m(x))$$

##  Additive Logistic Regression: A Statistical View of Boosting
다음으로 2000년에 Friedman 등이 AdaBoost algorithm을 통계적 관점에서 해석한 논문을 소개한다. 이 논문에서는 AdaBoost를 활용한 단계적 추정을 통해 최종 로지스틱 회귀 모델을 맞추었다. 
즉 AdaBoost Algorithm의 과정이 실제로 손실함수를 최소화하고 있음을 보여주었다.
<br/>

<div align="center">
<img src="/assets/figures/adaboost/adaboost2.png" 
title="Additive Logistic Regression: A Statistical View of Boosting" 
width="600"/>
</div>  ``

손실함수는 다음과 같이 표현할 수 있는데, 

$$L(y, F(x))=E(e^{-yF(x)})$$

이는 아래와 같은 포인트에서 최소화된다.

$$\frac{\partial E(e^{-yF(x)})}{\partial F(x)}=0$$

AdaBoost의 경우 $$y$$는 -1 또는 1만 될 수 있으므로 손실 함수는 다음과 같이 다시 작성할 수 있다.

$$E(e^{-yF(x)})=e^{F(x)}P(y=-1|x)+e^{-F(x)}P(y=1|x)$$

이를 $$F(x)$$에 대해 풀면, 아래와 같이 계산된다.

$$\frac{\partial E(e^{-yF(x)})}{\partial F(x)}=e^{F(x)}P(y=-1|x)-e^{-F(x)}P(y=1|x)=0$$

$$F(x)=\frac{1}{2}\log{\frac{P(y=1|x)}{P(y=-1|x)}}$$

또한 이 $$F(x)$$의 최적해로부터 로지스틱 모델을 유도할 수 있다.

$$P(y=-1|x)=\frac{e^{2F(x)}}{1+e^{2F(x)}}$$

만일 현재 추정치 $$F(x)$$와 개선된 추정치 $$F(x)+cf(x)$$가 있다면 고정된 $$x$$와 $$c$$에 대해 $$f(x)=0$$에 대한 2차식 $$L(y,F(x)+cf(X))$$을 얻을 수 있다.

$$L(y,F(x)+cf(X))=E(e^{-y(F(x)+cf(x))})$$ 

$$\approx E(e^{-yF(x)}(1-ycf(x)+(cyf(x))^2/2)))$$

$$=E(e^{-yF(x)}(1-ycf(x)+c^2/2))$$

$$\therefore f(x)=\mathit{argmin}_f E_w(1-ycf(x)+c^2/2|x)$$  


이때 $$E_w(1-ycf(x)+c^2/2 \mid x)$$는 가중된 조건부 기대값을 나타내며 각 데이터 포인트에 대한 가중치는 다음과 같이 계산된다.

$$w(x_i,y_i)= e^{-y_i F(x_i)}, i=1,\ldots,n$$

만약 $$c$$가 양수라면 가중된 조건부 기대값을 최소화하는 것은 $$E_w[yf(x)]$$를 최대화하는 것과 같다.
또한 $$y$$는 1 또는 -1의 값만 가질 수 있으므로 $$E_w[yf(x)]$$는 아래와 같이 쓸 수 있다.

$$E_w[yf(x)]=f(x)P_w(y=1|x)-f(x)P_w(y=-1|x)$$

$$ f(n)= \begin{cases}
1, & \mbox{if } P_w(y=1|x)>P_w(y=-1|x) \\
-1, & \mbox{if }\mbox{ otherwise.}
\end{cases}$$

이렇게 $$f(x)$$를 결정한 후 가중치 $$c$$는 $$L(y, F(x) + cf(x))$$를 직접 최소화하여 계산할 수 있다.

$$c=\mathit{argmin}_c E_w(e^{-cyf(x)})$$

$$\frac{\partial E(e^{-cyf(x)}}{\partial c}=E_w(-yf(x)e^{-cyf(x)})=0$$

$$E_w(1_{y \neq f(x)})e^c-E_w(1_{y=f(x)})e^{-c}$$

$$\epsilon$$을 잘못 분류된 케이스들의 가중합과 같이 두면,

$$\epsilon e^{c}-(1-\epsilon)e^{-c}=0$$

$$c=\frac{1}{2}\log{\frac{1-\epsilon}{\epsilon}}$$

즉, 약한 분류기의 정확도가 50% 미만일 경우 c는 음수가 된다.
또한 모델의 개선 후($$F(x)+cf(x)$$) 각 개별 데이터 포인트에 대한 가중치는 다음과  같다.

$$w(x_i,y_i)= e^{-y_i F(x_i)-c y_i f(x)}, i=1,\ldots,n$$

그러므로 각 데이터별 가중치는 다음과 같이 업데이트 된다.

$$w(x_i,y_i) \leftarrow w(x_i,y_i)e^{-cf(x_i)y_i}$$

이는 위에서 살펴본 AdaBoost Algorithm과 동일한 형태임을 알 수 있다. 따라서 AdaBoost를 지수 손실함수가 있는 모델의 각 반복 m에서 현재 추정치를 개선하기 위해 약한 분류기에 반복적으로 적합하여 
순방향 단계적 가산 모델로 해석하는 것이 합리적임을 알 수 있다.


$$w_{m+1}(x_i,y_i)=\frac{w_m(x_i,y_i)exp[-\theta_m y_i f_m(x_i)]}{Z_m}$$

$$\theta_m=\frac{1}{2}ln(\frac{1-\epsilon_m}{\epsilon_m})$$

## Code
Code 수행은 iris 데이터의 분류 문제에 AdaBoost 모델을 적용했다. 데이터는 7:3의 비율로 학습과 테스트 데이터를 나누었다.  

AdaBoost의 실제 활용에서는 다음과 같은 단계를 따른다.
1. 처음에 학습데이터 중 일부를 추출한다.
2. 선택되지 않은 나머지 학습데이터로 평가를 진행하면서(Validation Set) 선택된 학습데이터로 AdaBoost 모델을 반복적으로 학습한다.
3. 모델이 잘못 분류한 관측치에 더 높은 가중치를 할당하여, 다음 반복에서 이러한 관측치가 높은 분류 확률을 얻도록 학습한다.
4. 분류기의 정확도에 따라 각 반복에서 훈련된 분류기에 가중치를 할당한다. 더 정확한 분류기는 높은 가중치를 가진다.
5. 이 프로세스는 최종 모델이 학습데이터에 대해 모두 오류없이 적합하거나 지정된 수의 분류기(n_estimators)를 구축할 때까지 반복한다.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

adaboost_classifier = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
```
AdaBoost에서 가장 중요한 Hyper-parameter는 모델 훈련에 사용되는 약한 분류기인 base_estimator와 반복적으로 훈련할 약한 분류기의 수($$M$$)를 나타내는 n_estimators, 약한 분류기의 가중치에 기여하는 learning_rate이다. 
base_estimator, 즉 약한 분류기는 sklearn의 AdaBoost 모델의 default 설정 그대로 Decision Tree 모델을 사용하여 모델을 학습시킨 후, 이를 평가했다. 

```python
# Train Adaboost Classifer
model = adaboost_classifier.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
```
그 결과 88.88%의 분류 정확도를 얻을 수 있었다. 
```
Accuracy: 0.8888888888888888
```
다음으로 약한 분류기를 Support Vector Classifier로 활용한 뒤 최종 AdaBoost 모델을 구축했다.
```python
from sklearn.svm import SVC

svc=SVC(probability=True, kernel='linear')
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test, y_pred))
```
최종 AdaBoost 모델 구축 시 SVC를 약한 분류기(base estimator)로 활용했을 때 Decision Tree의 경우보다 Iris 데이터에 대하여 더 높은 성능을 보임을 알 수 있었다.

```
Accuracy: 0.9555555555555556
```
## 결론
이번 논문 구현 과제를 통해 AdaBoost 모델의 기본 개념을 숙지 할 수 있었다. 기본적으로 다양한 분류 모델들을 AdaBoost의 Base Estimator가 되는 약한 분류기로 사용할 수 있고 
이 약한 분류기의 실수를 반복적으로 수정하고 약한 분류기를 결합하여 정확도를 높이는 과정이기에 구현하기 쉬운 장점이 있다는 것을 알 수 있었다. 하지만 AdaBoost는 각 데이터 포인트에 완벽히 맞추려는
알고리즘의 특성상 outlier에 민감할 수 밖에 없다. 따라서 이러한 단점을 보완하는 Boosting 계열의 후속 연구들이 이어졌고 추후 나머지 알고리즘에 대해서도 공부해볼 계획이다.
추가적으로 데이터 사이즈가 클 경우 XGBoost에 비해 AdaBoost가 학습 속도가 느리다는 것을 이번 구현 과정에서 알 수 있었는데 학습 속도의 차이가 나타나는 정확한 이유에 대해서도 살펴볼 생각이다.

>**참고문헌**
1. Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
2. Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: a statistical view of boosting (with discussion and a rejoinder by the authors). The annals of statistics, 28(2), 337-407.``