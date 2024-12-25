import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import shutil
import time

def reset_logs():
    """
    重置日志目录：如果存在，则删除整个目录并重新创建
    """
    log_dir = f"logs_tf_keras_{int(time.time())}"  # 创建唯一日志目录
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除整个目录，而非仅尝试删除空目录
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    return log_dir

def iris_tf_keras():
    """
    使用 TensorFlow 和 TensorBoard 完成 Iris 数据集的分类任务
    """
    # 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # One-hot 编码
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 设置日志目录
    log_dir = reset_logs()
    print(f"TensorBoard 日志目录：{log_dir}")

    # 设置 TensorBoard 回调
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # 训练模型
    print("开始训练模型...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, callbacks=[tensorboard_callback])

    # 启动 TensorBoard
    print("启动 TensorBoard...")
    os.system(f'tensorboard --logdir {log_dir}')

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"TensorFlow 测试准确率：{test_acc}")

# 执行代码
iris_tf_keras()
