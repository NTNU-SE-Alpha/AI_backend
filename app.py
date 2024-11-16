from flask import Flask, request, jsonify, Response
import os
import json
import fitz  # PyMuPDF
import numpy as np
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt
from sentence_transformers import SentenceTransformer
import faiss
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import pathlib
from models import (
    db,
    Teacher,
    TeacherFiles,
    Conversation,
    Message,
    Student,
    TeacherFaiss,
)
from config import Config
from flask_migrate import Migrate
from datetime import datetime


app = Flask(__name__)
app.config.from_object(Config)


db.init_app(app)

migrate = Migrate(app, db)

jwt = JWTManager(app)


with app.app_context():
    db.create_all()


class AITeacher:
    def __init__(self):
        current_dir = pathlib.Path(__file__).parent.absolute()
        self.save_dir = os.path.join(current_dir, "saved_data")

        os.makedirs(self.save_dir, exist_ok=True)

        print(f"保存目錄路徑: {self.save_dir}")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key, max_tokens=4096, model_name="gpt-4o"
        )
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.system_context = None

    def summarize_text(self, text):
        messages = [
            SystemMessage(
                content="以下為教師傳給 AI 的問題，請用一段文字總結教師的問題，不要加上主詞"
            ),
            HumanMessage(content=text),
        ]

        try:
            summary = self.generate_response(messages)
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "無法生成摘要。"

    def load_conversation_history(self, conversation_uuid):
        conversation = Conversation.query.filter_by(uuid=conversation_uuid).first()
        if not conversation:
            return None, []

        messages = (
            Message.query.filter_by(conversation_id=conversation.id)
            .order_by(Message.sent_at)
            .all()
        )
        history = [(msg.id, msg.sender, msg.message, msg.sent_at) for msg in messages]
        return conversation, history

    def save_message(self, conversation_id, sender, message):
        new_message = Message(
            conversation_id=conversation_id, sender=sender, message=message
        )
        db.session.add(new_message)
        db.session.commit()

    def extract_text_from_pdf(self, pdf_path):
        """從 PDF 提取文本"""
        try:
            pdf_absolute_path = os.path.abspath(pdf_path)
            print(f"PDF 文件路徑: {pdf_absolute_path}")  # 調試信息

            if not os.path.exists(pdf_absolute_path):
                print(f"PDF 文件不存在: {pdf_absolute_path}")
                return ""

            doc = fitz.open(pdf_absolute_path)
            text = ""
            for page in doc:
                text += page.get_text()

            print(f"提取的文本長度: {len(text)}")  # 調試信息
            return text

        except Exception as e:
            print(f"提取 PDF 文本時發生錯誤: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return ""

    def build_faiss_index(self, text, save_name=None):
        try:
            # 分割句子並移除空白行
            sentences = [s.strip() for s in text.split("\n") if s.strip()]
            print(f"總句子數: {len(sentences)}")  # 調試信息

            if not sentences:
                print("警告：沒有找到有效的句子")
                return None, None

            # 生成嵌入向量
            print("開始生成嵌入向量...")  # 調試信息
            embeddings = self.model.encode(sentences)
            embeddings = embeddings.astype(np.float32)
            print(f"嵌入向量形狀: {embeddings.shape}")  # 調試信息

            # 建立索引
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            print(f"索引中的向量數: {index.ntotal}")  # 調試信息

            # 如果提供了保存名稱，則保存索引
            if save_name:
                save_success = self.save_faiss_index(index, sentences, save_name)
                if save_success:
                    print("索引保存成功")
                else:
                    print("索引保存失敗")

            return index, sentences

        except Exception as e:
            print(f"建立 FAISS 索引時發生錯誤: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return None, None

    def save_faiss_index(self, index, sentences, name="current"):
        """保存 FAISS 索引和對應的句子"""
        try:
            index_path = os.path.join(self.save_dir, f"{name}_index.faiss")
            sentences_path = os.path.join(self.save_dir, f"{name}_sentences.json")

            # 保存 FAISS 索引
            faiss.write_index(index, index_path)
            print(f"FAISS 索引已保存到: {index_path}")  # 調試信息

            # 保存對應的句子
            with open(sentences_path, "w", encoding="utf-8") as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)
            print(f"句子數據已保存到: {sentences_path}")  # 調試信息

            # 驗證文件是否確實被創建
            if os.path.exists(index_path) and os.path.exists(sentences_path):
                file_size_index = os.path.getsize(index_path)
                file_size_sentences = os.path.getsize(sentences_path)
                print(f"索引文件大小: {file_size_index} bytes")  # 調試信息
                print(f"句子文件大小: {file_size_sentences} bytes")  # 調試信息
                return True
            else:
                print("文件保存失敗，文件不存在")  # 調試信息
                return False

        except Exception as e:
            print(f"保存 FAISS 索引時發生錯誤: {str(e)}")
            import traceback

            print(traceback.format_exc())  # 打印完整的錯誤堆疊
            return False

    def load_faiss_index(self, name="current"):
        """載入 FAISS 索引和對應的句子"""
        try:
            index_path = os.path.join(self.save_dir, f"{name}_index.faiss")
            index = faiss.read_index(index_path)

            sentences_path = os.path.join(self.save_dir, f"{name}_sentences.json")
            with open(sentences_path, "r", encoding="utf-8") as f:
                sentences = json.load(f)

            return index, sentences
        except Exception as e:
            print(f"載入 FAISS 索引時發生錯誤: {e}")
            return None, None

    def search_rag(self, query, index, sentences, top_k=5):
        try:
            query_embedding = self.model.encode([query])
            distances, indices = index.search(query_embedding, top_k)
            return [sentences[i] for i in indices[0]]
        except Exception as e:
            print(f"Error during RAG search: {e}")
            return []

    def generate_response(self, messages):
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "抱歉，我無法處理您的請求。"


teacher = AITeacher()


@app.route("/login", methods=["POST"])
# 處理使用者登入
def login():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"message": "Username and password are required."}), 400

    # 先檢查是否為老師
    user = Teacher.query.filter_by(username=data["username"]).first()
    user_type = "teacher"

    if not user:
        # 如果不是老師，則檢查是否為學生
        user = Student.query.filter_by(username=data["username"]).first()
        user_type = "student"

    # 如果不是老師也不是學生
    if not user:
        return jsonify({"message": "Invalid username or password."}), 401

    # 如果帳號或密碼錯誤
    if not user.check_password(data["password"]):
        return jsonify({"message": "Invalid username or password."}), 401

    # 產生 JWT Token
    additional_claims = {"user_type": user_type, "user_id": user.id}
    access_token = create_access_token(
        identity=data["username"], additional_claims=additional_claims
    )

    # 回傳使用者的資料
    if user_type == "teacher":
        user_info = {"id": user.id, "username": user.username, "name": user.name}
    else:
        user_info = {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "course": user.course,
            "group": user.group_number,
        }

    return jsonify({"access_token": access_token, "user": user_info}), 200


@app.route("/start_conversation", methods=["POST"])
@jwt_required()
def start_conversation():
    claims = get_jwt()
    user_type = claims.get("user_type")
    user_id = claims.get("user_id")
    if not user_type or not user_id:
        return jsonify({"message": "Invalid token."}), 400

    if user_type == "teacher":
        user = Teacher.query.get(user_id)
        if not user:
            return jsonify({"message": "User not found."}), 404

    new_conversation = Conversation(teacher_id=user_id)
    db.session.add(new_conversation)
    db.session.commit()
    return jsonify({"uuid": new_conversation.uuid})


@app.route("/chat/<string:conversation_uuid>", methods=["POST"])
@jwt_required()
def chat(conversation_uuid):
    claims = get_jwt()
    user_type = claims.get("user_type")
    user_id = claims.get("user_id")
    if not user_type or not user_id:
        return jsonify({"message": "Invalid token."}), 400

    if user_type == "teacher":
        user = Teacher.query.get(user_id)
        if not user:
            return jsonify({"message": "User not found."}), 404

    if not conversation_uuid:
        return jsonify({"message": "The UUID of conversation is required."}), 400

    conversation = Conversation.query.filter_by(uuid=conversation_uuid).first()

    if conversation is None:
        return jsonify({"message": "The UUID of conversation is invalid."}), 400

    if conversation.teacher_id != user_id:
        return jsonify({"message": "Not authorized."}), 401

    data = request.json
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify(
            {"message": "The UUID of conversation and the user input are required."}
        ), 400

    if "file_id" in data:
        file = TeacherFiles.query.filter_by(id=data["file_id"]).first()

        if file is None:
            return jsonify({"message": "file_id is invalid."}), 400

        file_content = teacher.extract_text_from_pdf(file.path)

        if not file_content:
            return jsonify({"message": "Unable to read file."}), 400

        faiss_file = TeacherFaiss.query.filter_by(file_id=data["file_id"]).first()

        if faiss_file is not None:
            index, sentences = teacher.load_faiss_index(data["file_id"])
            if index is None or sentences is None:
                faiss_index = TeacherFaiss.query.get(faiss_file.id)
                db.session.delete(faiss_index)
                db.session.commit()
                return jsonify({"message": "Unable to read faiss file"}), 400
        else:
            index, sentences = teacher.build_faiss_index(file_content, data["file_id"])

            if index is None or sentences is None:
                print("建立索引失敗，程式結束。")
                return

            faiss_index = TeacherFaiss(file_id=data["file_id"])
            db.session.add(faiss_index)
            db.session.commit()

        teacher.system_context = f"""您是一位AI教學助手。
以下是課程內容的摘要：{file_content[:1000]}...
請基於上述內容來回答問題。如果需要引入新的例子或故事，請確保與原始課程內容保持關聯。"""

        relevant_context = teacher.search_rag(user_input, index, sentences)

        context = "\n".join(relevant_context)

    else:
        teacher.system_context = ""
        context = ""

    conversation, conversation_history = teacher.load_conversation_history(
        conversation_uuid
    )
    if not conversation:
        return jsonify({"message": "The UUID of conversation is invalid."}), 400

    if len(conversation_history) == 0:
        conversation.summary = teacher.summarize_text(user_input)
        db.session.commit()

    messages = [
        SystemMessage(content=teacher.system_context),
        SystemMessage(content=f"相關上下文：\n\n{context}")
    ]

    for _, q, a, _ in conversation_history:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))

    messages.append(HumanMessage(user_input))

    answer = teacher.generate_response(messages)
    conversation_history.append((user_input, answer))

    teacher.save_message(conversation.id, "user", user_input)
    teacher.save_message(conversation.id, "assistant", answer)

    return jsonify({"answer": answer})


@app.route("/history/<string:conversation_uuid>", methods=["GET"])
@jwt_required()
def get_history(conversation_uuid):
    claims = get_jwt()
    user_type = claims.get("user_type")
    user_id = claims.get("user_id")
    if not user_type or not user_id:
        return jsonify({"message": "Invalid token."}), 400

    if user_type == "teacher":
        user = Teacher.query.get(user_id)
        if not user:
            return jsonify({"message": "User not found."}), 404

    if not conversation_uuid:
        return jsonify({"message": "The UUID of conversation is required."}), 400

    conversation = Conversation.query.filter_by(uuid=conversation_uuid).first()

    if conversation is None:
        return jsonify({"message": "The UUID of conversation is invalid."}), 400

    if conversation.teacher_id != user_id:
        return jsonify({"message": "Not authorized."}), 401

    conversation, conversation_history = teacher.load_conversation_history(
        conversation_uuid
    )

    if not conversation:
        return jsonify({"message": "The UUID of conversation is invalid."}), 400

    formatted_history = [
        {
            "id": id,
            "sender": sender,
            "message": msg,
            "sent_at": sent_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for id, sender, msg, sent_at in conversation_history
    ]
    return jsonify({"uuid": conversation_uuid, "history": formatted_history})


@app.route("/list_conversations", methods=["GET"])
@jwt_required()
def list_conversations():
    claims = get_jwt()
    user_type = claims.get("user_type")
    user_id = claims.get("user_id")
    if not user_type or not user_id:
        return jsonify({"message": "Invalid token."}), 400

    if user_type == "teacher":
        user = Teacher.query.get(user_id)
        if not user:
            return jsonify({"message": "User not found."}), 404

    conversations = Conversation.query.filter_by(teacher_id=user_id).all()

    conversation_list = [
        {
            "uuid": conversation.uuid,
            "summary": conversation.summary
            if conversation.summary
            else "No summary available",
            "created_at": conversation.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for conversation in conversations
    ]

    return jsonify({"conversations": conversation_list})


if __name__ == "__main__":
    if not os.path.exists(teacher.save_dir):
        os.makedirs(teacher.save_dir)
    app.run(debug=True)