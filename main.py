import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from  sklearn.model_selection import train_test_split

# 读取Excel文件中的数据
df = pd.read_excel("./data/pinpu.xlsx")  # 替换为您的Excel文件路径

# 分离特征和目标变量
X = df.iloc[:, :-1].values  # 获取所有行，除了最后一列的所有列（即特征）
y = df.iloc[:, -1].values  # 获取所有行，只有最后一列（即标签）
print(X.shape)

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

# 创建神经网络模型
model = Sequential()

# 添加一维卷积层，假设我们想要64个过滤器，每个过滤器的大小为3
model.add(Conv1D(filters=1, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))

# 添加一维池化层，用于降维
model.add(MaxPooling1D(pool_size=2))

# 展平层，将多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())

# 添加全连接层
model.add(Dense(64, activation='relu'))

# 输出层，使用softmax激活函数进行多分类
model.add(Dense(5, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 输出模型结构
model.summary()

# 训练模型
model.fit(X, y_one_hot, epochs=200, batch_size=32)

# 保存模型
model.save('my_conv_nn_model.h5')





# 读取测试Excel文件中的数据
#test_df = pd.read_excel("./data/test_gan_20.xlsx")  # 替换为您的测试Excel文件路径

# 分离测试集的特征和目标变量
# test_X = test_df.iloc[:, :-1].values  # 获取所有行，除了最后一列的所有列（即特征）
# test_y = test_df.iloc[:, -1].values  # 获取所有行，只有最后一列（即标签）

# 将测试标签转换为数值型
#test_y = np.array([label_dict[label] for label in test_y])

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



