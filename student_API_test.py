import subprocess
import json

while True:
    curl_command = input("請輸入 curl 命令: ")

    # 使用 subprocess 來執行 curl 命令，並確保使用 utf-8 編碼處理
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True, encoding='utf-8')
    print(str(result))
    # 檢查 curl 是否執行成功
    if result.returncode == 0:
        # 嘗試將回傳的內容解析為 JSON 格式
        try:
            response_data = json.loads(result.stdout)
            
            # 檢查是否有 outline，並輸出
            if 'answer' in response_data:
                print("answer generated:")
                print(response_data['answer'])
            else:
                print("answer not found in response.")
        
        except json.JSONDecodeError as e:
            print("Failed to parse the response as JSON:")
            print(result.stdout)
            print(f"JSONDecodeError: {e}")

    else:
        print(f"Error: Curl command failed with return code {result.returncode}")
        print("Error output:")
        print(result.stderr)
