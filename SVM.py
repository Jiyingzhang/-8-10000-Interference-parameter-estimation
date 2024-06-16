import numpy as np
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,LSTM,Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from  sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            for i in range(num_samples):
                if y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)

class MulticlassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.models = {}

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        num_classes = len(self.unique_classes)

        for i in range(num_classes):
            for j in range(i+1, num_classes):
                class1 = self.unique_classes[i]
                class2 = self.unique_classes[j]

                binary_X = X[(y == class1) | (y == class2)]
                binary_y = y[(y == class1) | (y == class2)]
                binary_y = np.where(binary_y == class1, 1, -1)

                svm_model = SVM()
                svm_model.fit(binary_X, binary_y)
                self.models[(class1, class2)] = svm_model

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)

        for i in range(num_samples):
            scores = np.zeros(len(self.unique_classes))
            for (class1, class2), model in self.models.items():
                prediction = model.predict(X[i])
                if prediction == 1:
                    scores[class1] += 1
                else:
                    scores[class2] += 1
            predictions[i] = np.argmax(scores)

        return predictions

# # 准备频谱数据和对应的标签
# X = np.random.rand(100, 1024)  # 假设有100个频谱数据
# y = np.random.randint(0, 5, 100)  # 随机生成对应的标签（5类）
# print(y)
# 读取Excel文件中的数据
df = pd.read_excel("D:/桌面/PINPU/data.xlsx",header=None)  # 替换为您的Excel文件路径
# df = pd.read_excel("D:/桌面/cnn0/cnn0/data/pinpu.xlsx",header=None)# 替换为您的Excel文件路径
# df = pd.read_excel("./pinpu.xlsx",header=None)  # 替换为您的Excel文件路径

# 分离特征和目标变量
X = df.iloc[:, :-1].values  # 获取所有行，除了最后一列的所有列（即特征）
y = df.iloc[:, -1].values  # 获取所有行，只有最后一列（即标签）
print(X.shape)
print(X)
# # 实例化一个 MinMaxScaler 对象
# scaler = MinMaxScaler()
#
# # 对特征进行归一化
# X_normalized = scaler.fit_transform(X)

# # 如果标签是文本形式的类别，我们需要先将其转换为数值型
# label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}  # 根据您的标签替换映射
# y = np.array([label_dict[label] for label in y])
X,X_test,y,y_test=train_test_split(X,y,test_size=0.2, stratify=y)
print(y)

# 如果标签是分类的，并且我们想要使用one-hot编码
# num_classes = len(label_dict)  # 获取类别数
y_one_hot = to_categorical(y, num_classes=5)
# 创建多类别SVM模型并训练
multi_svm = MulticlassSVM()
multi_svm.fit(X, y)

# 在训练集上评估模型
predictions = multi_svm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print("模型在训练集上的准确率为:", accuracy)
# 计算混淆矩阵
cm = confusion_matrix(y_test, predictions)

# 打印混淆矩阵
print(cm)