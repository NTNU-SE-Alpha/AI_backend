# AI 教師與學生助手使用說明

本文檔詳細說明了兩個 Python 程式 `AI_teacher_temp.py` 和 `AI_student_temp.py` 的功能和注意事項。這兩個程式旨在協助教師和學生進行教學互動，利用自然語言處理和機器學習技術來提高學習體驗。

---

## 目錄

- [環境設定與依賴項目](#環境設定與依賴項目)
- [AI_teacher_temp.py 詳細說明](#ai_teacher_temppy-詳細說明)
  - [程式概述](#程式概述)
  - [主要模組和函式](#主要模組和函式)
  - [使用方法](#使用方法)
  - [注意事項](#注意事項)
- [AI_student_temp.py 詳細說明](#ai_student_temppy-詳細說明)
  - [程式概述](#程式概述-1)
  - [主要模組和函式](#主要模組和函式-1)
  - [使用方法](#使用方法-1)
  - [注意事項](#注意事項-1)
- [常見問題與解決方案](#常見問題與解決方案)

---

## 環境設定與依賴項目

在運行這兩個程式之前，需要確保開發環境已經正確配置。

### 環境要求

- Python 版本：`Python 3.7` 或更高版本
- 作業系統：Windows、macOS 或 Linux

### 主要依賴項目

- `os`：標準庫，用於與作業系統互動。
- `json`：標準庫，用於處理 JSON 數據。
- `fitz`：PyMuPDF，用於從 PDF 中提取文本。
- `dotenv`：用於加載 `.env` 文件中的環境變數。
- `numpy`：數值計算庫，用於處理數組和矩陣。
- `sentence_transformers`：用於生成句子嵌入。
- `faiss`：Facebook AI Similarity Search，用於高效的相似度搜索。
- `langchain_openai`：用於與 OpenAI 的 GPT-4 模型互動。
- `pathlib`：標準庫，用於處理路徑操作。
- `datetime`：標準庫，用於處理日期和時間。

### 安裝依賴項

請在命令行中運行以下指令以安裝所有必要的依賴項：

```bash
pip install PyMuPDF python-dotenv numpy sentence-transformers faiss-cpu langchain-openai
```

> **注意**：`faiss` 可能需要特殊的安裝方式，具體可參考其[官方安裝指南](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)。

### 配置環境變數

創建一個 `.env` 文件，並添加您的 OpenAI API 金鑰：

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## AI_teacher_temp.py 詳細說明

### 程式概述

`AI_teacher_temp.py` 是一個為教師設計的 AI 助手，旨在協助教師從 PDF 教材中提取內容，建立 FAISS 索引，並與教師進行互動，回答教師的問題或建議。該程式能夠：

- 從指定的 PDF 文件中提取文本內容。
- 將文本內容分割成句子並生成嵌入向量。
- 建立和保存 FAISS 索引以供快速相似度搜索。
- 使用 OpenAI 的 GPT-4 模型生成回答。
- 保存對話歷史並生成教學總結。

### 主要模組和函式

#### 1. `AITeacher` 類

此類包含了 AI 教師助手的所有主要功能。

##### 初始化方法 `__init__(self)`

- **功能**：初始化 AI 教師助手，包括設置保存目錄、加載模型和配置 OpenAI API。
- **操作**：
  - 設置保存數據的目錄 `saved_data`。
  - 加載 OpenAI API 金鑰和初始化 GPT-4 模型。
  - 加載 `SentenceTransformer` 模型以生成句子嵌入。
  - 初始化對話歷史和系統上下文。

##### `save_faiss_index(self, index, sentences, name="current")`

- **功能**：保存 FAISS 索引和對應的句子列表。
- **參數**：
  - `index`：FAISS 索引對象。
  - `sentences`：與索引對應的句子列表。
  - `name`：保存的文件名稱前綴。
- **操作**：
  - 將索引保存為 `.faiss` 文件。
  - 將句子列表保存為 `.json` 文件。
  - 驗證文件是否成功保存。

##### `load_faiss_index(self, name="current")`

- **功能**：加載已保存的 FAISS 索引和句子列表。
- **參數**：
  - `name`：要加載的文件名稱前綴。
- **返回值**：索引對象和句子列表。

##### `build_faiss_index(self, text, save_name=None)`

- **功能**：建立 FAISS 索引。
- **參數**：
  - `text`：要索引的文本內容。
  - `save_name`：如果提供，則將索引和句子列表保存到指定名稱的文件中。
- **操作**：
  - 將文本內容按行分割成句子，並移除空白行。
  - 生成句子的嵌入向量。
  - 建立 FAISS 索引。
  - 選擇性地保存索引和句子列表。

##### `extract_text_from_pdf(self, pdf_path)`

- **功能**：從 PDF 文件中提取文本。
- **參數**：
  - `pdf_path`：PDF 文件的路徑。
- **返回值**：提取的文本內容。

##### `save_lesson_to_faiss(self, lesson_content)`

- **功能**：將教學內容和對話歷史保存到 FAISS 索引。
- **參數**：
  - `lesson_content`：教學內容的文本。
- **操作**：
  - 將教學內容和對話歷史合併。
  - 建立並保存新的 FAISS 索引。

##### `search_rag(self, query, index, sentences, top_k=5)`

- **功能**：使用檢索增強生成（RAG）進行相似度搜索，找到與查詢相關的句子。
- **參數**：
  - `query`：查詢字符串。
  - `index`：FAISS 索引對象。
  - `sentences`：句子列表。
  - `top_k`：返回最相關的前 `k` 個句子。
- **返回值**：相關的句子列表。

##### `generate_response(self, messages)`

- **功能**：使用 GPT-4 模型生成回答。
- **參數**：
  - `messages`：包含系統消息、用戶消息和 AI 消息的列表。
- **返回值**：生成的回答字符串。

##### `summarize_lesson(self)`

- **功能**：總結整個教學過程。
- **返回值**：教學總結字符串。

##### `save_conversation_history(self, filename="conversation_history.json")`

- **功能**：將對話歷史保存為 JSON 文件。
- **參數**：
  - `filename`：保存的文件名稱。

##### `run(self, pdf_path)`

- **功能**：主程序，執行 AI 教師助手的整個流程。
- **參數**：
  - `pdf_path`：教材 PDF 文件的路徑。
- **操作**：
  - 提取教材內容。
  - 建立並保存 FAISS 索引。
  - 進行與教師的互動。
  - 保存對話歷史和生成教學總結。

#### 2. `main()` 函式

- **功能**：程式的入口點，初始化 `AITeacher` 並運行主程序。
- **操作**：
  - 獲取當前目錄並設置 PDF 文件的路徑。
  - 初始化 `AITeacher` 對象並調用 `run()` 方法。

### 使用方法

1. **確保環境配置正確**：安裝所有必要的依賴項，並設置 OpenAI API 金鑰。

2. **準備教材 PDF 文件**：將您的教材文件命名為 `teaching_resources.pdf`，並放置在與 `AI_teacher_temp.py` 相同的目錄下。

3. **運行程式**：

   ```bash
   python AI_teacher_temp.py
   ```

4. **進行互動**：程式將提示您輸入教師的問題或建議，按照提示進行操作。

5. **查看結果**：對話歷史和 FAISS 索引將被保存到 `saved_data` 目錄中，教學總結將在程式結束時顯示。

### 注意事項

- **PDF 文件格式**：確保 PDF 文件格式正確，文本內容清晰，以便程式能夠正確提取文本。

- **OpenAI API 金鑰**：請確保您的 API 金鑰有效且有足夠的調用配額。

- **模型性能**：生成嵌入和建立索引可能需要一定的時間，具體取決於教材內容的長度。

- **錯誤處理**：程式中已經包含了一些基本的錯誤處理，但在某些情況下可能仍需要手動調試。

---

## AI_student_temp.py 詳細說明

### 程式概述

`AI_student_temp.py` 是一個為學生設計的 AI 助手，旨在協助學生理解教材內容，並根據教師的教學風格和要求進行互動。該程式能夠：

- 載入教師與 AI 助手的對話歷史，理解教師的教學期望。
- 載入教材內容，為學生提供準確的答案。
- 使用 OpenAI 的 GPT-4 模型生成回答。
- 與學生進行連續的對話互動。

### 主要模組和函式

#### 1. `AIStudentAssistant` 類

此類包含了 AI 學生助手的所有主要功能。

##### 初始化方法 `__init__(self)`

- **功能**：初始化 AI 學生助手，包括加載模型和配置 OpenAI API。
- **操作**：
  - 設置對話歷史和教材的文件路徑。
  - 加載 OpenAI API 金鑰和初始化 GPT-4 模型。
  - 加載 `SentenceTransformer` 模型以生成句子嵌入。
  - 初始化系統上下文。

##### `load_conversation_records(self)`

- **功能**：載入並預處理教師與 AI 助手的對話紀錄。
- **操作**：
  - 從 JSON 文件中載入對話紀錄。
  - 清理數據，過濾無關信息。
  - 生成句子的嵌入向量。

##### `load_teaching_material(self)`

- **功能**：載入教材內容並提取文本。
- **操作**：
  - 從 PDF 文件中提取文本內容。
  - 儲存提取的教材文本。

##### `search_relevant_context(self, query, top_k=5)`

- **功能**：根據學生的查詢，在對話紀錄中搜索相關的上下文。
- **參數**：
  - `query`：學生的問題或輸入。
  - `top_k`：返回最相關的前 `k` 個句子。
- **返回值**：相關的句子列表。

##### `generate_response(self, messages)`

- **功能**：使用 GPT-4 模型生成回答。
- **參數**：
  - `messages`：包含系統消息、學生消息和 AI 消息的列表。
- **返回值**：生成的回答字符串。

##### `interact_with_student(self)`

- **功能**：與學生進行互動的主流程。
- **操作**：
  - 載入對話紀錄和教材內容。
  - 進行與學生的對話，根據學生的輸入生成回答。
  - 在對話中引入相關的上下文和教材內容。

#### 2. `main()` 函式

- **功能**：程式的入口點，初始化 `AIStudentAssistant` 並開始互動。

### 使用方法

1. **確保環境配置正確**：安裝所有必要的依賴項，並設置 OpenAI API 金鑰。

2. **準備文件**：

   - **對話歷史**：確保 `conversation_history.json` 文件存在，且包含教師與 AI 助手的對話紀錄。
   - **教材 PDF 文件**：將您的教材文件命名為 `teaching_resources.pdf`，並放置在與 `AI_student_temp.py` 相同的目錄下。

3. **運行程式**：

   ```bash
   python AI_student_temp.py
   ```

4. **進行互動**：程式將提示您以學生的身份輸入問題，按照提示進行操作。

5. **查看結果**：AI 助手將根據您的輸入和教材內容提供回答。

### 注意事項

- **文件完整性**：請確保 `conversation_history.json` 和 `teaching_resources.pdf` 文件存在且格式正確。

- **OpenAI API 金鑰**：請確保您的 API 金鑰有效且有足夠的調用配額。

- **模型性能**：生成嵌入和搜索相關內容可能需要一定的時間。

- **錯誤處理**：程式中已經包含了一些基本的錯誤處理，但在某些情況下可能仍需要手動調試。

---

## 常見問題與解決方案

**Q1：運行程式時提示 OpenAI API 金鑰未設置？**

- **A**：請確保已在 `.env` 文件中設置了 `OPENAI_API_KEY`，或者直接在程式中提供您的 API 金鑰。

**Q2：程式無法提取 PDF 文件的文本內容？**

- **A**：請檢查 PDF 文件是否存在，路徑是否正確，並確保文件不是加密的。如果文件較大，提取可能需要較長時間。

**Q3：生成回答時出現錯誤或響應緩慢？**

- **A**：這可能是由於網絡問題或 OpenAI API 的限制。請檢查您的網絡連接，並確保您的 API 金鑰有足夠的配額。

**Q4：程式提示無法載入對話紀錄或教材？**

- **A**：請確認相關的文件是否存在於正確的目錄中，並且文件格式正確（對話紀錄為 JSON 格式，教材為 PDF 格式）。

**Q5：如何修改程式以適應自己的需求？**

- **A**：您可以根據自己的需求修改程式的參數，如更換教材文件、調整搜索相關內容的數量、修改生成回答的風格等。

---
