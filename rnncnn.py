import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, SimpleRNN, TimeDistributed

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,LSTM,Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from  sklearn.model_selection import train_test_split
import openpyxl
# 读取Excel文件中的数据
# df = pd.read_excel("D:/桌面/PINPU/data.xlsx",header=None)  # 替换为您的Excel文件路径
df = pd.read_excel("./pinpu.xlsx",header=None)# 替换为您的Excel文件路径
df['Column1']=df['Column1'].astype(int)

# 分离特征和目标变量
X = df.iloc[:, :-1].values  # 获取所有行，除了最后一列的所有列（即特征）
y = df.iloc[:, -1].values  # 获取所有行，只有最后一列（即标签）
print(X.shape)
print(X)

# # 如果标签是文本形式的类别，我们需要先将其转换为数值型
# label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}  # 根据您的标签替换映射
# y = np.array([label_dict[label] for label in y])
X,X_test,y,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

# 如果标签是分类的，并且我们想要使用one-hot编码
# num_classes = len(label_dict)  # 获取类别数
y_one_hot = to_categorical(y, num_classes=5)

# 为了使用一维卷积层，我们需要将特征数据的维度增加一个，使其成为(samples, timesteps, features)
# 假设每个样本有60个特征，我们将这些特征视作序列中的时间步长
# 这里我们假设每个时间步长只有1个特征，因此增加一个维度(60, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)


# 假设输入数据的形状是 (batch_size, time_steps, input_dim)
time_steps = 1024  # 序列长度
input_dim = 1  # 每个时间步的特征维度
num_classes = 5  # 假设的输出类别数量
lstm_units = 64  # RNN层的单元数
l2_reg_strength = 0.01

# 创建Sequential模型
model = Sequential()

# 添加Conv1D层，假设我们使用大小为3的卷积核和64个输出通道
model.add(Conv1D(256, 6, activation='relu', input_shape=(time_steps, input_dim)))
model.add(MaxPooling1D(pool_size=8))
model.add(Conv1D(128, 3, activation='relu', input_shape=(time_steps, input_dim)))
model.add(MaxPooling1D(pool_size=4))

# 如果需要在RNN之前减少维度，可以添加一个Flatten层
#model.add(Flatten())  # 注意：在RNN之前通常不需要展平，除非你想要完全忽视时间步的顺序
# model.add(Flatten())
# 添加RNN层，这里以SimpleRNN为例，你也可以使用LSTM或GRU
#model.add(LSTM(rnn_units, activation='relu'))
model.add(LSTM(lstm_units, activation='relu'))

model.add(Dropout(0.3))

# 如果你希望在RNN之后再次进行卷积，可以使用TimeDistributed包装器
# model.add(TimeDistributed(Conv1D(32, 3, activation='relu')))

# 添加全连接层作为输出层
# model.add(Dense(800, activation='relu'))
model.add(Dense(500, activation='relu'))
# model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型结构
model.summary()

# 训练模型
model.fit(X, y_one_hot, epochs=200, batch_size=32)



# 对测试特征数据进行维度增加
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 如果需要，将测试标签转换为one-hot编码（通常评估模型时不需要）
# test_y_one_hot = to_categorical(test_y, num_classes=num_classes)

# 评估模型在测试集上的性能
loss, accuracy = model.evaluate(X_test, to_categorical(y_test, num_classes=5))
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 预测测试集的结果
predictions = model.predict(X_test)

# 如果需要，将预测结果从one-hot编码转换回原始类别标签
predicted_classes = np.argmax(predictions, axis=1)

# 打印预测结果
print("Predicted classes:", predicted_classes)

# 如果你想查看测试集的真实标签和预测标签的对比
print("True labels:", y_test)


# 假设 y_test 是测试集的真实标签，y_pred 是模型对测试集的预测标签
#y_test = [...]  # 真实标签列表
#y_pred = [...]  # 预测标签列表

# 计算混淆矩阵
cm = confusion_matrix(y_test, predicted_classes)

# 打印混淆矩阵
print(cm)