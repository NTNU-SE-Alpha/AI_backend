# AI_backend

## Basic Requirements
**使用到的技術**
- 文件轉成 RAG
- text splitter 設定的大小不要超過 window size 的 1/4
- 使用 FAISS 進行 RAG 檢索


**會用到的功能**
- 寫一個 function，將老師與 AI 對話得出的**最後教學內容**，放進學生 AI 的 system prompt 裡(需要有一個給 AI 人設的 prompt 模板，可以使用 Claude Prompt generator)。
- 將老師與 AI 對話的內容轉成 RAG，並跟老師上傳給 AI 對話的內容包在一起，作為學生 AI 的 RAG。
- 上傳資料只能限定 `pdf`
- 給後端 function 來傳老師的文件
- 提供一鍵總結教學內容的功能

## Advance
- 根據文件內容提供多元化問題(EX: 產生跟文件內容相關的問題、想讓教學重點放在哪個部分)
