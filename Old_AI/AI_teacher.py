import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import pathlib
from datetime import datetime

# 設置環境變數以禁用 tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 載入環境變數
load_dotenv()

class AITeacher:
    def __init__(self):
        # 獲取當前腳本的絕對路徑
        current_dir = pathlib.Path(__file__).parent.absolute()
        self.save_dir = os.path.join(current_dir, "saved_data")
        
        # 創建保存目錄
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"保存目錄路徑: {self.save_dir}")  # 調試信息
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            max_tokens=4096,
            model_name="gpt-4"
        )
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.conversation_history = []
        self.system_context = None

    def save_faiss_index(self, index, sentences, name="current"):
        """保存 FAISS 索引和對應的句子"""
        try:
            index_path = os.path.join(self.save_dir, f"{name}_index.faiss")
            sentences_path = os.path.join(self.save_dir, f"{name}_sentences.json")
            
            # 保存 FAISS 索引
            faiss.write_index(index, index_path)
            print(f"FAISS 索引已保存到: {index_path}")  # 調試信息
            
            # 保存對應的句子
            with open(sentences_path, 'w', encoding='utf-8') as f:
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
            with open(sentences_path, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
            
            return index, sentences
        except Exception as e:
            print(f"載入 FAISS 索引時發生錯誤: {e}")
            return None, None

    def build_faiss_index(self, text, save_name=None):
        """建立 FAISS 索引並選擇性地保存"""
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

    def save_lesson_to_faiss(self, lesson_content):
        """保存教學記錄到 FAISS"""
        try:
            conversation_text = "\n".join([f"老師: {q}\nAI: {a}" for q, a in self.conversation_history])
            all_content = lesson_content + "\n" + conversation_text
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            index, sentences = self.build_faiss_index(all_content, save_name=f"lesson_{timestamp}")
            return index, sentences
        except Exception as e:
            print(f"保存教學記錄到 FAISS 時發生錯誤: {e}")
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

    def summarize_lesson(self):
        try:
            messages = [SystemMessage(content="請總結以下對話的重點：")]
            for q, a in self.conversation_history:
                messages.append(HumanMessage(content=q))
                messages.append(AIMessage(content=a))
            return self.generate_response(messages)
        except Exception as e:
            print(f"Error summarizing lesson: {e}")
            return "無法生成教學總結。"

    def save_conversation_history(self, filename="conversation_history.json"):
        try:
            filepath = os.path.join(self.save_dir, filename)
            # 按照指定格式構建對話列表
            conversation_list = []
            for q, a in self.conversation_history:
                conversation_list.append(f"老師: {q}")
                conversation_list.append(f"AI: {a}")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(conversation_list, f, ensure_ascii=False, indent=4)
            print(f"對話歷史已保存到: {filepath}")
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    def run(self, pdf_path):
        print("\n=== 開始執行 AI 教師系統 ===")
        print(f"使用的 PDF 文件: {pdf_path}")
        
        # 讀取課程內容
        lesson_content = self.extract_text_from_pdf(pdf_path)
        if not lesson_content:
            print("未能提取課程內容，請檢查 PDF 路徑。")
            return

        # 建立索引並保存
        print("\n建立 FAISS 索引...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index, sentences = self.build_faiss_index(lesson_content, save_name="current")
        if index is None or sentences is None:
            print("建立索引失敗，程式結束。")
            return

        print("\n=== 系統初始化完成，開始互動 ===")
        
        # 儲存初始系統上下文
        self.system_context = f"""您是一位AI教學助手。
以下是課程內容的摘要：{lesson_content[:1000]}...
請基於上述內容來回答問題。如果需要引入新的例子或故事，請確保與原始課程內容保持關聯。"""

        while True:
            user_input = input("請輸入教師的問題或建議：").strip()
            if not user_input:
                print("輸入無效，請重新輸入。")
                continue

            # 搜尋相關內容
            relevant_context = self.search_rag(user_input, index, sentences)
            context = "\n".join(relevant_context)

            # 準備完整的對話歷史
            messages = [
                SystemMessage(content=self.system_context),
                SystemMessage(content=f"相關上下文：\n\n{context}")
            ]
            
            # 添加之前的對話歷史
            for q, a in self.conversation_history:
                messages.append(HumanMessage(content=q))
                messages.append(AIMessage(content=a))
            
            # 添加當前問題
            messages.append(HumanMessage(content=user_input))

            # 生成回應
            answer = self.generate_response(messages)

            # 更新對話歷史
            self.conversation_history.append((user_input, answer))
            
            print(f"AI 回應：{answer}\n")

            # 確認是否繼續
            if input("是否繼續互動？(yes/no): ").strip().lower() != "yes":
                break

        # 儲存對話歷史
        self.save_conversation_history()

        # 生成教學總結
        lesson_summary = self.summarize_lesson()
        print(f"\n教學總結：{lesson_summary}\n")

        # 在結束時保存最終的索引
        final_index, final_sentences = self.save_lesson_to_faiss(lesson_content)
        if final_index and final_sentences:
            print(f"對話歷史和 FAISS 索引已保存到 {self.save_dir} 目錄。")
        else:
            print("儲存 FAISS 索引時出現問題。")

def main():
    # 使用絕對路徑
    current_dir = pathlib.Path(__file__).parent.absolute()
    pdf_path = os.path.join(current_dir, "teaching_resources.pdf")
    
    print(f"程式運行目錄: {current_dir}")
    print(f"PDF 文件路徑: {pdf_path}")
    
    teacher = AITeacher()
    teacher.run(pdf_path)

if __name__ == "__main__":
    main()
