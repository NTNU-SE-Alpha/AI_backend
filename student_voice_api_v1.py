from flask import Flask, request, jsonify
import threading
import openai
import mysql.connector
import os
import uuid
import json
from pydub import AudioSegment
import librosa
import numpy as np

app = Flask(__name__)

# 設置 OpenAI API 密鑰
openai.api_key = openai_api_key = "sk-iRhZc3RJiGLWLL7kzE8sU7kXd22e7RhrndDIkZdnDLT3BlbkFJstYK5Xu_TJWhEddQII8tlCCAuicKPIRx1kQZZeDgEA"  # 請使用有效的 OpenAI API Key

# 数据库配置
db_config = {
    'user': 'david',
    'password': 'Benben921023',
    'host': 'localhost',
    'database': 'your_db_name',
}

# 任務翻譯結果和鎖
translations_lock = threading.Lock()

# 動態範圍壓縮
def apply_compression(audio):
    compressed_audio = audio.compress_dynamic_range(
        threshold=-30.0, ratio=4.0, attack=5.0, release=50.0
    )
    return compressed_audio

# 計算動態 EQ 濾波參數
def calculate_filter_parameters(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    high_pass_freq = spectral_centroid * 0.5
    low_pass_freq = spectral_centroid + spectral_bandwidth
    return high_pass_freq, low_pass_freq

# 應用動態 EQ 濾波
def apply_dynamic_eq(audio, high_pass_freq, low_pass_freq):
    audio = audio.high_pass_filter(high_pass_freq)
    audio = audio.low_pass_filter(low_pass_freq)
    return audio

# 音頻增強
def enhance_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    compressed_audio = apply_compression(audio)
    high_pass_freq, low_pass_freq = calculate_filter_parameters(file_path)
    eq_audio = apply_dynamic_eq(compressed_audio, high_pass_freq, low_pass_freq)
    louder_audio = eq_audio + 16
    normalized_audio = louder_audio.normalize()
    enhanced_file_path = file_path.replace(".mp3", "_enhanced.mp3")
    normalized_audio.export(enhanced_file_path, format="mp3")
    return enhanced_file_path

@app.route('/api/tasks/<task_id>/audio', methods=['POST'])
def upload_audio(task_id):
    # 解析 JSON 內容
    data = request.json
    order = int(data.get('order', 0))
    file_path = data.get('file_path')

    if not file_path:
        return jsonify({'error': '缺少文件路徑'}), 400

    # 增強音質
    enhanced_file_path = enhance_audio(file_path)

    # 使用 OpenAI Whisper API 處理音頻文件
    with open(enhanced_file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file, language="zh")

    transcription = response['text']

    # 打印轉錄內容
    print(f"Order {order} Transcription: {transcription}")

    # 存储翻译结果到数据库
    store_transcription(task_id, order, transcription)

    # 刪除臨時文件
    os.remove(file_path)
    os.remove(enhanced_file_path)

    # 更新 summary 欄位
    update_summary(task_id)

    return jsonify({'task_id': task_id, 'order': order, 'text': transcription}), 200

@app.route('/api/tasks/<task_id>/complete', methods=['POST'])
def complete_task(task_id):
    with translations_lock:
        all_transcriptions = get_all_transcriptions(task_id)
        concatenated_text = ''.join([t['transcription'] for t in sorted(all_transcriptions, key=lambda x: x['order'])])

        # GPT 生成大綱和摘要
        outline = generate_outline(concatenated_text)
        conversation_summary = generate_conversation_summary(concatenated_text)

        # 存入 MySQL
        store_analysis(task_id, outline, conversation_summary)

        return jsonify({'outline': outline, 'conversation_summary': conversation_summary}), 200

@app.route('/api/tasks/<task_id>/transcription', methods=['GET'])
def get_transcription(task_id):
    # 從資料庫中獲取轉錄結果
    all_transcriptions = get_all_transcriptions(task_id)
    transcription_list = [t['transcription'] for t in sorted(all_transcriptions, key=lambda x: x['order'])]
    return jsonify({'task_id': task_id, 'transcriptions': transcription_list}), 200

# GPT 生成大綱
def generate_outline(text):
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "system", "content": "請生成中文大綱。"}, {"role": "user", "content": text}],
        max_tokens=2000, temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip()

# GPT 生成對話總結
def generate_conversation_summary(text):
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "system", "content": "請總結對話內容。"}, {"role": "user", "content": text}],
        max_tokens=2000, temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip()

# 存储分析結果到 MySQL
def store_analysis(task_id, outline, conversation_summary):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = """
    INSERT INTO analysis (task_id, analysis_text, chat)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE analysis_text = VALUES(analysis_text), chat = VALUES(chat)
    """
    cursor.execute(query, (task_id, outline, conversation_summary))
    conn.commit()
    cursor.close()
    conn.close()

# 存储翻译結果到 MySQL
def store_transcription(task_id, order, transcription):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = """
    INSERT INTO translations (task_id, `order`, transcription)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE transcription = VALUES(transcription)
    """
    cursor.execute(query, (task_id, order, transcription))
    conn.commit()
    cursor.close()
    conn.close()

# 更新 summary 欄位
def update_summary(task_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query_select = "SELECT transcription FROM translations WHERE task_id = %s ORDER BY `order`"
    cursor.execute(query_select, (task_id,))
    transcriptions = cursor.fetchall()
    combined_transcription = ' '.join([t[0] for t in transcriptions])
    query_update = "UPDATE translations SET summary = %s WHERE task_id = %s"
    cursor.execute(query_update, (combined_transcription, task_id))
    conn.commit()
    cursor.close()
    conn.close()

# 获取所有翻译結果
def get_all_transcriptions(task_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    query = "SELECT `order`, transcription FROM translations WHERE task_id = %s"
    cursor.execute(query, (task_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

if __name__ == '__main__':
    app.run(debug=True)
