import os
import json
import PyPDF2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# 載入 .env 檔案中的環境變數
load_dotenv()

# 設定環境變數，解決 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AIStudentAssistant:
    def __init__(self):
        self.conversation_sentences_path = "conversation_history.json"
        self.teaching_material_path = "teaching_resources.pdf"

        # 初始化 OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("無法找到 OPENAI_API_KEY，請設置正確的環境變數或直接提供 API 金鑰。")

        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            max_tokens=4096,
            model_name="gpt-4-turbo"
        )
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.system_context = None
        self.sentences = None
        self.sentence_embeddings = None
        self.teaching_content = None

    def load_conversation_records(self):
        """載入對話紀錄並進行預處理
        """
        try:
            print("正在載入對話紀錄...")
            with open(self.conversation_sentences_path, 'r', encoding='utf-8') as f:
                raw_sentences = json.load(f)
            
            # 過濾並清理句子
            cleaned_sentences = []
            for sentence in raw_sentences:
                # 過濾掉檔案路徑、數字和短字串
                if (len(str(sentence).strip()) > 5 and 
                    not str(sentence).startswith('K:\\') and
                    not str(sentence).endswith('.png') and
                    not str(sentence).isdigit()):
                    cleaned_sentences.append(str(sentence).strip())
            
            # 計算嵌入向量（只計算一次）
            self.sentences = cleaned_sentences
            self.sentence_embeddings = self.model.encode(cleaned_sentences)
            
            print(f"成功載入並處理了 {len(cleaned_sentences)} 條對話紀錄")
            return True
        except Exception as e:
            print(f"載入對話紀錄時發生錯誤: {e}")
            return False

    def load_teaching_material(self):
        """載入教材內容並變成文字格式
        """
        try:
            print("正在載入教材...")
            with open(self.teaching_material_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            
            self.teaching_content = text.strip()
            print("成功載入教材")
            return True
        except Exception as e:
            print(f"載入教材時發生錯誤: {e}")
            return False

    def search_relevant_context(self, query, top_k=5):
        """搜尋相關上下文
        """
        try:
            if not self.sentences or self.sentence_embeddings is None:
                return []
            
            # 計算查詢的嵌入向量
            query_embedding = self.model.encode(query)
            
            # 計算相似度
            similarities = [
                (sentence, float(query_embedding @ embedding))
                for sentence, embedding in zip(self.sentences, self.sentence_embeddings)
            ]
            
            # 排序並返回最相關的句子
            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            return [sentence for sentence, similarity in sorted_similarities[:top_k] if similarity > 0.3]
            
        except Exception as e:
            print(f"搜尋過程中發生錯誤: {e}")
            return []

    def generate_response(self, messages):
        """生成 AI 回應
        """
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"生成回應時發生錯誤: {e}")
            return "抱歉，我現在無法處理您的請求。"

    def interact_with_student(self):
        """與學生互動
        """
        if not self.load_conversation_records() or not self.load_teaching_material():
            print("無法載入對話紀錄或教材，請確保檔案存在且格式正確。")
            return

        print("\n=== AI 助教已準備就緒，可以開始互動 ===")
        print("資料載入完成，可以開始互動嘍！")
        
        self.system_context = (
            "您是一位 AI 助教，以台式繁體中文回答問題，根據對話紀錄中老師的期待方式以及教材內容來回答學生的問題或引導討論。"
            "對話紀錄中包含老師希望的回答方式，而教材是主要的內容基礎，請根據這些參考資料進行回答。"
            "請使用適合教學的語氣和方式來回答。"
        )

        conversation_history = []

        while True:
            try:
                student_input = input("\n學生: ").strip()
                if not student_input:
                    print("輸入無效，請重新輸入。")
                    continue

                relevant_context = self.search_relevant_context(student_input)
                context = "相關內容：\n" + "\n".join(relevant_context) if relevant_context else "找不到直接相關的內容。"

                messages = [
                    SystemMessage(content=self.system_context),
                    SystemMessage(content=f"這些是老師先前的對話紀錄，他希望課堂進行的一些特殊需求就寫在這邊：{context}"),
                    SystemMessage(content=f"這個是今天要用到的教材，請依照裡面的內容帶討論或是回答學生問題：{self.teaching_content}")
                ]
                
                for q, a in conversation_history[-3:]:  # 只保留最近的3次對話
                    messages.append(HumanMessage(content=q))
                    messages.append(AIMessage(content=a))
                
                messages.append(HumanMessage(content=student_input))

                answer = self.generate_response(messages)
                conversation_history.append((student_input, answer))
                
                print(f"\nAI 助教: {answer}")

                continue_chat = input("\n是否繼續對話？(yes/no): ").strip().lower()
                if continue_chat != "yes":
                    break
                    
            except KeyboardInterrupt:
                print("\n\n程式已終止")
                break


def main():
    assistant = AIStudentAssistant()
    assistant.interact_with_student()

if __name__ == "__main__":
    main()
