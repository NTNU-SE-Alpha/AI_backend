import os
import threading
import time
import datetime
import random
import re
import warnings
import json
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from flask import Flask, request, Response
import openai
import pickle
import mysql.connector

app = Flask(__name__)
warnings.filterwarnings("ignore")  # 忽略所有警告

# 设置上传文件的存储路径
UPLOAD_FOLDER = os.path.expanduser('~/uploads')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置 OpenAI API 密钥（使用环境变量）
openai.api_key = "sk-iRhZc3RJiGLWLL7kzE8sU7kXd22e7RhrndDIkZdnDLT3BlbkFJstYK5Xu_TJWhEddQII8tlCCAuicKPIRx1kQZZeDgEA" # 請替換為您的 API 密鑰

# 文件类型与大小限制
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# 模型初始化
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 每个 user_id 对应一个 UserSession 实例
user_sessions = {}
session_lock = threading.Lock()  # 锁，用于保护 user_sessions 字典的并发访问

# 日志类，每个用户有自己的日志
class UserLog:
    def __init__(self, user_id):
        self.user_id = user_id
        self.log_file = f'logs/{user_id}.txt'
        self.lock = threading.Lock()
        # 确保日志目录存在
        if not os.path.exists('logs'):
            os.makedirs('logs')

    def log(self, text):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{current_time}] {text}\n")

# 用户会话类，包含每个用户的数据结构
class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.index = None
        self.paragraphs = []
        self.log = UserLog(user_id)
        self.initialize_faiss_index()

    def initialize_faiss_index(self):
        if self.index is None:
            embedding_dim = 384
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.log.log("已初始化 FAISS 索引")

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 检查文件大小是否在允许范围内
def check_file_size(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size <= MAX_FILE_SIZE

# 简单的文本切割函数
def split_text(text, max_length=15):
    sentences = re.split(r'(?<=[。！？])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    processed_sentences = []
    for sentence in sentences:
        if len(sentence) > max_length:
            words = re.findall(r'\S.{0,' + str(max_length-1) + r'}\b', sentence)
            processed_sentences.extend(words)
        else:
            processed_sentences.append(sentence)
    return processed_sentences

# 功能：处理文字并建立 FAISS 索引
def add_text_to_faiss(session, text):
    split_sentences = split_text(text, max_length=15)
    for sentence in split_sentences:
        new_embedding = model.encode([sentence])
        session.index.add(new_embedding)
        session.paragraphs.append(sentence)
    session.log.log(f"文本已成功切割并加入 FAISS 索引，共 {len(split_sentences)} 段。")
    remove_similar_entries(session)
    # 在此处保存索引
    save_faiss_index(session, app.config['UPLOAD_FOLDER'])

# 功能：处理文件并建立 FAISS 索引
def add_file_to_faiss(session, file_path):
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    clean_text = text.replace("\n", " ")
    split_sentences = split_text(clean_text, max_length=15)
    for sentence in split_sentences:
        new_embedding = model.encode([sentence])
        session.index.add(new_embedding)
        session.paragraphs.append(sentence)
    session.log.log(f"文件内容已成功切割并加入 FAISS 索引，共 {len(split_sentences)} 段。")
    remove_similar_entries(session)
    # 在此处保存索引
    save_faiss_index(session, app.config['UPLOAD_FOLDER'])

# 功能：删除相似数据，只保留一个
def remove_similar_entries(session, threshold=0.5):
    if len(session.paragraphs) == 0:
        return
    embeddings = np.array([session.index.reconstruct(i) for i in range(len(session.paragraphs))])
    D, I = session.index.search(embeddings, 10)
    to_remove = set()
    for i in range(len(I)):
        for j in range(1, len(I[i])):
            if D[i][j] < threshold:
                to_remove.add(I[i][j])
    remaining_indices = [i for i in range(len(session.paragraphs)) if i not in to_remove]
    new_embeddings = np.array([session.index.reconstruct(i) for i in remaining_indices])
    new_index = faiss.IndexFlatL2(session.index.d)
    new_index.add(new_embeddings)
    session.index = new_index
    session.paragraphs = [session.paragraphs[i] for i in remaining_indices]
    session.log.log(f"已删除相似数据，剩余 {len(session.paragraphs)} 条数据")
    # 在此处保存索引
    save_faiss_index(session, app.config['UPLOAD_FOLDER'])
#生成大綱

def generate_natural_language_outline(session):
    if not session.paragraphs:
        session.log.log("目前 FAISS 索引中没有数据。")
        return "目前没有可用的数据生成大纲。"
    
    max_paragraphs = min(250, len(session.paragraphs))  # 限制最大段落数为250
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(session.paragraphs)

    if len(session.paragraphs) > 250:
        num_kmeans = 130
        num_random = 120
    else:
        num_kmeans = len(session.paragraphs) // 2
        num_random = len(session.paragraphs) - num_kmeans

    num_clusters = min(num_kmeans, X.shape[0])
    if num_clusters < 2:
        num_clusters = 1

    selected_kmeans_paragraphs = []
    
    if X.shape[0] >= num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        for cluster_idx in range(num_clusters):
            cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_idx]
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(X[cluster_indices] - kmeans.cluster_centers_[cluster_idx], axis=1))]
            selected_kmeans_paragraphs.append(closest_idx)

    remaining_indices = list(set(range(len(session.paragraphs))) - set(selected_kmeans_paragraphs))

    # 将剩余段落分成20组，每组均匀随机抽取
    num_groups = 20
    group_size = max(1, len(remaining_indices) // num_groups)  # 确保每组至少有一个段落

    # 将剩余的段落划分为 20 组
    grouped_indices = [remaining_indices[i:i + group_size] for i in range(0, len(remaining_indices), group_size)]

    # 在每组中随机选择一个段落
    random_selected_paragraphs = []
    for group in grouped_indices:
        if group:  # 确保组内有段落
            random_selected_paragraphs.append(random.choice(group))

    selected_paragraphs = selected_kmeans_paragraphs + random_selected_paragraphs

    selected_texts = [session.paragraphs[idx] for idx in selected_paragraphs]
    combined_text = "\n".join(selected_texts[:max_paragraphs])

    prompt = f"请基于以下数据生成一个简短的大纲：\n\n{combined_text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个帮助生成大纲的助手。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )

    outline = response['choices'][0]['message']['content'].strip()
    session.log.log("已生成大纲")
    return outline

# 功能：清空 FAISS 索引
def clear_faiss_index(session):
    session.index = None
    session.paragraphs = []
    session.initialize_faiss_index()
    session.log.log("FAISS 索引已清空，所有数据已重置")
    # 在此处保存索引
    save_faiss_index(session, app.config['UPLOAD_FOLDER'])

# 功能：保存 FAISS 索引
def save_faiss_index(session, directory):
    if session.index is not None:
        index_file = os.path.join(directory, f"{session.user_id}_faiss.index")
        faiss.write_index(session.index, index_file)
        # 保存 paragraphs
        paragraphs_file = os.path.join(directory, f"{session.user_id}_paragraphs.pkl")
        with open(paragraphs_file, 'wb') as f:
            pickle.dump(session.paragraphs, f)
        session.log.log(f"FAISS 索引已保存到 {index_file}")
    else:
        session.log.log("FAISS 索引为空，无法保存")

# 功能：加载 FAISS 索引
def load_faiss_index(session, directory):
    index_file = os.path.join(directory, f"{session.user_id}_faiss.index")
    paragraphs_file = os.path.join(directory, f"{session.user_id}_paragraphs.pkl")
    if os.path.exists(index_file) and os.path.exists(paragraphs_file):
        try:
            session.index = faiss.read_index(index_file)
            with open(paragraphs_file, 'rb') as f:
                session.paragraphs = pickle.load(f)
            session.log.log(f"已从 {index_file} 载入 FAISS 索引")
        except (EOFError, pickle.UnpicklingError) as e:
            session.log.log(f"读取索引文件时发生错误：{e}，将重新初始化索引")
            session.index = None
            session.paragraphs = []
            session.initialize_faiss_index()
    else:
        session.log.log("没有找到对应的 FAISS 索引，初始化新索引")
        session.initialize_faiss_index()
def save_to_sql(session, connection, cursor):
    if session.index is not None:
        try:
            # 序列化 FAISS 索引
            faiss_data = faiss.serialize_index(session.index)
            # 序列化段落列表
            paragraphs_data = pickle.dumps(session.paragraphs)

            # SQL 查询
            query = """
                REPLACE INTO user_data (user_id, faiss_index, paragraphs)
                VALUES (%s, %s, %s)"""
            user_id = session.user_id

            # 执行查询
            cursor.execute(query, (user_id, faiss_data, paragraphs_data))
            connection.commit()

            # 检查影响的行数
            if cursor.rowcount > 0:
                session.log.log(f"FAISS 索引已成功保存到 SQL，影响行数：{cursor.rowcount}")
            else:
                session.log.log("SQL 执行成功，但没有影响行数，可能未更新任何数据。")

        except Exception as e:
            session.log.log(f"SQL 保存失败: {str(e)}")

    else:
        session.log.log("FAISS 索引为空，无法保存到 SQL")

# API 路由
@app.route('/main', methods=['POST'])
def main():
    data = request.form  # 使用 form 来获取 multipart/form-data
    user_id = data.get('user_id')
    action = int(data.get('action', 0))

    if not user_id:
        message = {"error": "必须提供 user_id"}
        return Response(
            response=json.dumps(message, ensure_ascii=False),
            mimetype='application/json',
            status=400
        )

    with session_lock:
        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id)

    session = user_sessions[user_id]
    session.log.log("程序启动")

    # 在每次請求中動態建立資料庫連線
    connection = mysql.connector.connect(
        host="localhost",             
        user="david",                 
        password="Benben921023",      
        database="faiss_paragraph_db"
    )
    cursor = connection.cursor()

    try:
        # 读取该用户的 FAISS 索引（如果存在）
        load_faiss_index(session, app.config['UPLOAD_FOLDER'])

        if action == 1:
            # 上傳檔案處理...
            pass
        elif action == 2:
            # 上傳文字處理...
            pass
        elif action == 3:
            session.log.log("用户清空数据")
            clear_faiss_index(session)
            message = {"message": "数据已清空", "outline": "NONE"}
            return Response(
                response=json.dumps(message, ensure_ascii=False),
                mimetype='application/json',
                status=200
            )

        elif action == 4:
            session.log.log("用户输出大纲")
            outline = generate_natural_language_outline(session)
            message = {"message": "大纲生成成功", "outline": outline}
            return Response(
                response=json.dumps(message, ensure_ascii=False),
                mimetype='application/json',
                status=200
            )
        elif action == 5:
            session.log.log("用户保存大纲到数据库")
            save_to_sql(session, connection, cursor)
            outline= generate_natural_language_outline(session)
            message = {"message": "大纲上傳成功", "outline": outline}
            return Response(
                response=json.dumps(message, ensure_ascii=False),
                mimetype='application/json',
                status=200
            )

        else:
            session.log.log("无法识别用户选择")
            message = {"error": "无效的操作选择"}
            return Response(
                response=json.dumps(message, ensure_ascii=False),
                mimetype='application/json',
                status=400
            )
    except Exception as e:
        session.log.log(f"發生錯誤: {str(e)}")
        message = {"error": f"內部錯誤: {str(e)}"}
        return Response(
            response=json.dumps(message, ensure_ascii=False),
            mimetype='application/json',
            status=500
        )
    finally:
        # 關閉游標和連接
        cursor.close()
        connection.close()
        
if __name__ == '__main__':
    connection = mysql.connector.connect(
    host="localhost",             # 或者使用正确的 host
    user="david",                 # 替换为你的用户名
    password="Benben921023",      # 替换为你的密码
    database="faiss_paragraph_db"
    )
    cursor = connection.cursor()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
    # 关闭连接
    cursor.close()
    connection.close()
    
    
#https://chatgpt.com/share/671cb644-50b4-8010-bf6a-4c5774f6d5bf    API格式以及sql格式
