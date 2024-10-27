import os
import pickle
import warnings
import json
from flask import Flask, request, Response
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import faiss
import mysql.connector  # 用于连接 MySQL 数据库
import openai
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore")  # 忽略所有警告

app = Flask(__name__)

# 初始化 SentenceTransformer 模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 使用一个句子嵌入模型

openai.api_key = openai_api_key = "sk-iRhZc3RJiGLWLL7kzE8sU7kXd22e7RhrndDIkZdnDLT3BlbkFJstYK5Xu_TJWhEddQII8tlCCAuicKPIRx1kQZZeDgEA"  # 請使用有效的 OpenAI API Key

# 数据库连接配置，从环境变量中获取
db_config = {
    'host': "localhost",
    'user': "david",      # 替换为您的 MySQL 用户名
    'password': "Benben921023",  # 替换为您的 MySQL 密码
    'database': "conversation_db"
}

db_config_faiss = {
    'host': "localhost",
    'user': "david",      # 替换为您的 MySQL 用户名
    'password': "Benben921023",  # 替换为您的 MySQL 密码
    'database': "faiss_paragraph_db"
}

# FAISS 搜索函数
def search(query, index, model, sentences, top_k=50):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    results = [sentences[i] for i in indices[0]]
    return results

# 使用 K-means 选择段落
def select_paragraphs_with_kmeans(sentences, n_clusters=5, top_k=50):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(sentences)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    cluster_centers = kmeans.cluster_centers_

    # 使用 FAISS 找到每个聚类中心最近的段落
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray().astype('float32'))
    _, closest = index.search(cluster_centers.astype('float32'), 1)

    selected_paragraphs = [sentences[int(i)] for i in closest.ravel()[:top_k]]

    return selected_paragraphs

# 使用 GPT 生成大纲
def generate_outline(results):
    context = " ".join(results)
    messages = [
        {"role": "system", "content": "你是一個專業助手，幫助用戶生成大綱。"},
        {
            "role": "user",
            "content": f"根據以下資料生成一個簡單的大綱，這個大綱是用於給予 GPT 提示用的，用於輔助 GPT 的回覆更加具有精準度，生成一個相對完整的大綱，允許去修改它:\n\n{context}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=800
    )
    return response['choices'][0]['message']['content'].strip()

# 使用 GPT 生成最終回答
def generate_answer(outline, question):
    messages = [
        {
            "role": "system",
            "content": "你是一個專業助手，提供有幫助且具體的回答。請你顧及他人的情緒，多使用流行用語以及表情符號"
        },
        {
            "role": "user",
            "content": f"大綱: {outline}\n\n問題: {question}\n回答:"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=2500
    )
    return response['choices'][0]['message']['content'].strip()

# 從 MySQL 數據庫加載 FAISS 資料


def load_faiss_data_from_db(user_id):
    try:
        connection = mysql.connector.connect(**db_config_faiss)
        cursor = connection.cursor()
        query = "SELECT faiss_index, paragraphs FROM user_data WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        if result:
            faiss_index_blob, paragraphs_blob = result
            # 将 BLOB 转换为 numpy 数组，再传递给 faiss.deserialize_index
            faiss_index = faiss.deserialize_index(np.frombuffer(faiss_index_blob, dtype=np.uint8))
            # 反序列化段落列表
            paragraphs = pickle.loads(paragraphs_blob)
            return faiss_index, paragraphs
        else:
            return None, None
    except mysql.connector.Error as err:
        print(f"MySQL 错误：{err}")
        return None, None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()



# 保存對話記錄到數據庫
def save_conversation_to_db(user_id, question, answer):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = "INSERT INTO conversation_history (user_id, question, answer, timestamp) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (user_id, question, answer, datetime.now()))
        connection.commit()
    except mysql.connector.Error as err:
        print(f"MySQL 錯誤：{err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

# 定義 Flask 路由
@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    content = data.get('content')
    user_id = data.get('user_id')

    if not content or not user_id:
        message = {'error': '缺少 content 或 user_id'}
        return Response(
            response=json.dumps(message, ensure_ascii=False),
            mimetype='application/json',
            status=400
        )

    # 加載 FAISS 資料
    index, paragraphs = load_faiss_data_from_db(user_id)
    if index is None or paragraphs is None:
        message = {'error': '未找到該用戶的 FAISS 資料'}
        return Response(
            response=json.dumps(message, ensure_ascii=False),
            mimetype='application/json',
            status=404
        )

    # 執行 FAISS 搜索
    faiss_results = search(content, index, model, paragraphs)
    top_faiss_results = faiss_results[:30]  # 前 30 個最相關句子

    # 使用 K-means 選擇段落
    kmeans_results = select_paragraphs_with_kmeans(paragraphs, n_clusters=5, top_k=70)

    # 合併結果
    combined_results = top_faiss_results + kmeans_results

    # 生成大綱
    outline = generate_outline(combined_results)

    # 生成最終回答
    answer = generate_answer(outline, content)

    # 保存對話記錄到數據庫
    save_conversation_to_db(user_id, content, answer)

    # 返回 JSON 格式的回答
    message = {'answer': answer}
    return Response(
        response=json.dumps(message, ensure_ascii=False),
        mimetype='application/json',
        status=200
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

#https://chatgpt.com/share/671cb6e6-6b28-8010-9abc-5017591d223f   API格式以及sql格式