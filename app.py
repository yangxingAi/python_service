
from flask import Flask, request,session
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pymysql
from flask_cors import CORS
import creat_table
import knn
import random_forest

from gevent import pywsgi

app = Flask(__name__)

CORS(app)


def connect_mysql():
    connection = pymysql.connect(
        host='10.16.48.219',
        user='root',
        password='111111',
        database='medical',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection


def get_data(connection, table_name):
    query = f"select * from {table_name}"
    data = pd.read_sql(query, connection)
    connection.close()
    return data


# 通过路径  /pca/yangxing
@app.route('/pca', methods=['POST'])
def pca():
    connection = connect_mysql()
    # 查询数据库获取连接
    data = get_data(connection, "Diabetes")
    # 获取需要降维的特征
    # features = data.drop(['Case_ID'],axis=1)
    params = request.get_json()
    features = data[params]
    # 数据预处理：处理缺失值
    features = features.fillna(0)
    # 特征标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # 计算协方差矩阵
    covariance_matrix = np.cov(scaled_features.T)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # 选择主成分数量
    # 这里假设选择保留前10个主成分
    n_components = len(params) - 1
    # 降维
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(scaled_features)
    result = pd.DataFrame(reduced_features)
    explained_variance_ratio = pca.explained_variance_ratio_
    contribution_list = explained_variance_ratio.tolist()
    return [result.to_dict(), contribution_list]


@app.route("/knn", methods=['POST'])
def get_knn():
    return knn.knn()


@app.route("/featureCreate", methods=['POST'])
def feature_create():
    param = request.get_json()

    new_table = creat_table.create_table(param)
    # 连接到数据库
    connection = connect_mysql()

    # 创建游标对象
    cursor = connection.cursor()

    # 从数据库中读取数据
    query = f"SELECT * FROM {new_table} limit 15"
    cursor.execute(query)

    # 获取查询结果
    rows = cursor.fetchall()

    return rows


@app.route("/randomForest", methods=['POST'])
def fe():
    param = request.get_json()
    # {'tableName1': 'cardio_train', 'tableName2': 'stroke', 'aiName': 'randomForest', 'runParams': ['Case_ID', 'AGE', 'SEX']}
    return random_forest.random_forest(param)


@app.route("/randomForest1", methods=['POST'])
def fe1():
    param = request.get_json()
    # {'tableName1': 'cardio_train', 'tableName2': 'stroke', 'aiName': 'randomForest', 'runParams': ['Case_ID', 'AGE', 'SEX']}
    return random_forest.random_forest1(param)







if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
