import io

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import seaborn as sns

sns.set_style('darkgrid')

df = pd.read_csv('Graduate Admission/Admission_Predict_Ver1.1.csv')
app = Flask(__name__)
model = load_model('models/model.keras')
CORS(app)

with open('models/scaler.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)


# 加载Keras模型
def load_keras_model(data):
    print('model: ' + str(scaler_loaded))
    data = scaler_loaded.transform(data)
    return model.predict(np.array(data))


def load_other_model(models_path, data):
    job_model = joblib.load(f"models/{models_path}_model.joblib")
    print('model: ' + str(job_model))

    print('model: ' + str(scaler_loaded))

    data_scaled = scaler_loaded.transform(data)
    # 使用加载的模型进行预测
    prediction = job_model.predict(data_scaled)
    return prediction


@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json(force=True)

        input_data = [int(data['gre_score']), int(data['toefl_score']), int(data['university_rating']),
                      float(data['sop']), float(data['lor']), float(data['cgpa']), int(data['research'])]
        model_choice = data['model_choice']

        data = np.expand_dims(input_data, axis=0)

        # 根据 model_choice 调用相应的模型预测函数
        if model_choice == 'Keras':
            prediction = load_keras_model(data)
        else:
            prediction = load_other_model(model_choice, data)

        # prediction = load_keras_model(data)
        # prediction = load_other_model(model_choice, data)

        # r2 = r2_score(prediction, input_data)

        print('input_data: ' + str(input_data))
        print('model_choice: ' + str(model_choice))
        print('prediction: ' + str(prediction[0]))
        print()

        if model_choice == 'Keras':
            return jsonify({
                'model_choice': model_choice,
                'prediction': str(prediction[0][0])
            })
        else:
            return jsonify({
                'model_choice': model_choice,
                'prediction': str(prediction[0])
            })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/generateImage', methods=['GET'])
def generateImage1():
    dataframe = pd.read_csv('Graduate Admission/Admission_Predict_Ver1.1.csv')

    # 设置图表大小
    dataframe.drop(columns='Serial No.', inplace=True)
    plt.figure(figsize=(15, 10))

    # 筛选数值类型的列
    numeric_columns = dataframe.select_dtypes(include=['number']).columns

    # 循环绘制每个数值列的直方图
    for i, col in enumerate(numeric_columns):
        plt.subplot(4, 2, i + 1)
        sns.histplot(dataframe[col], kde=True, color='green')
        plt.xlabel(col)

    # 调整布局并展示图表
    plt.tight_layout()
    # 保存图表到字节流并返回
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


@app.route('/generateImage2', methods=['GET'])
def generateImage2():
    dataframe = pd.read_csv('Graduate Admission/Admission_Predict_Ver1.1.csv')
    cols = df.drop(labels='Serial No.', axis=1)

    # 计算相关性矩阵
    corr = cols.corr()

    # 创建一个用于掩盖矩阵上三角的掩码
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # 设置绘图风格和图像大小
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(9, 7))
        # 绘制带掩码的热图，设置正方形格子，注释，格式化数字，线宽和颜色映射
        ax = sns.heatmap(corr, mask=mask, square=True, annot=True, fmt='0.2f', linewidths=.8, cmap="hsv")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


@app.route('/generateImage3', methods=['GET'])
def generateImage3():
    dataframe = pd.read_csv('Graduate Admission/Admission_Predict_Ver1.1.csv')
    cols = dataframe.drop(labels='Serial No.', axis=1)

    # 计算相关性矩阵
    corr = cols.corr()
    plt.figure(figsize=(20, 15))
    plt.rcParams['axes.facecolor'] = "#e6ffed"
    plt.rcParams['figure.facecolor'] = "#e6ffed"
    g = sns.pairplot(data=cols, hue='Research', markers=["^", "v"], palette='inferno')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
