# Classifier and Dimensionality Reduction

这是人工智能实验，简单的手写数字分类识别和人脸图像去噪。

## Classifier

### Navie Bayes

under the above independence assumptions, the conditional distribution over the class variable {\displaystyle C} C is:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2ca793ac821823121f3d5f508269d945a58acf11)
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/9dd841d7c36e6d7449bea439ef99e8138810870d)

I use the simplist `Parameter estimation` to calculate the probility P(Xi | Ck)

### Linear Classifiers with L2 Regularization parameter

using the matlab function `quadprog`

Optimize: min( (Xw-y)^2 + lambda*w'*w )= (Xw-y)'*(Xw-y) + lambda*w'*w = w'*(X'X+lambda)*w -2y'Xw +y'y

solution: `(X'*X + lambda)\X'*y
