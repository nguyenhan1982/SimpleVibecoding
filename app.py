import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt')
        model_type = data.get('model', 'default')
        api_key = data.get('apiKey')
        context = data.get('context', '')
        task_type = data.get('taskType', 'code') # code, ideas, prompt

        # Build prompt in backend
        if task_type == 'ideas':
            topic = data.get('topic', '')
            count = data.get('count', 5)
            full_prompt = f"""Hãy tạo {count} ý tưởng ứng dụng web CHI TIẾT và THỰC TẾ liên quan đến chủ đề "{topic}". 
            Yêu cầu:
            - Sắp xếp từ đơn giản đến phức tạp.
            - Mỗi ý tưởng phải có tên gọi sáng tạo và một đoạn mô tả chức năng đầy đủ (khoảng 3-4 câu), nêu rõ giá trị cốt lõi và cách người dùng tương tác.
            - Định dạng mỗi ý tưởng bắt đầu bằng: "Ý tưởng [số]: [Tên ứng dụng] - [Mô tả chi tiết]".
            - Viết bằng tiếng Việt.
            Hãy tập trung vào tính khả thi và trải nghiệm người dùng."""
            
        elif task_type == 'prompt':
            desc = data.get('desc', '')
            current_key = api_key if api_key else 'YOUR_API_KEY'
            
            if 'groq' in model_type:
                model_id = "qwen/qwen3-32b" if 'qwen' in model_type else "moonshotai/kimi-k2-instruct-0905"
                max_tokens = 40960 if 'qwen' in model_type else 16384
                top_p = 0.95 if 'qwen' in model_type else 1
                model_name = "Qwen3-32b (Groq)" if 'qwen' in model_type else "KimiK2 (Groq)"
                reasoning = ',\n                             "reasoning_effort": "default"' if 'qwen' in model_type else ''
                
                full_prompt = f"""Dựa trên mô tả: "{desc}". Hãy tạo một "Prompt Chuẩn" TRÌNH BÀY CHÍNH XÁC theo cấu trúc 4 mục sau:
                1. Tạo ứng dụng "[Tên ứng dụng]". Ứng dụng có chức năng "[Mô tả chức năng chính]".
                2. Đầu vào của ứng dụng là: [Liệt kê các đầu vào cần thiết].
                3. Đầu ra của ứng dụng là: [Mô tả kết quả mong muốn].
                4. Chức năng "[Tên chức năng chính]" được mô hình "{model_name}" xử lý. Kết nối với mô hình "{model_name}" thông qua giao thức sau: 
                curl "https://api.groq.com/openai/v1/chat/completions" \\
                  -X POST \\
                  -H "Content-Type: application/json" \\
                  -H "Authorization: Bearer {current_key}" \\
                  -d '{{
                         "messages": [
                           {{
                             "role": "user",
                             "content": ""
                           }}
                         ],
                         "model": "{model_id}",
                         "temperature": 0.6,
                         "max_completion_tokens": {max_tokens},
                         "top_p": {top_p},
                         "stream": true{reasoning},
                         "stop": null
                       }}'
                
                Yêu cầu: Trả về văn bản thuần theo đúng cấu trúc trên, không thêm lời dẫn giải."""
                
            elif 'cerebras' in model_type:
                model_name_c = 'Llama3.1-8b (Cerebras)' if 'llama3.1-8b' in model_type else 'Gpt-oss-120b (Cerebras)'
                model_id_c = 'llama3.1-8b' if 'llama3.1-8b' in model_type else 'gpt-oss-120b'
                max_tokens_c = 8192 if 'llama3.1-8b' in model_type else 32768
                reasoning_c = '' if 'llama3.1-8b' in model_type else '\n                      "reasoning_effort": "medium",'
                
                full_prompt = f"""Dựa trên mô tả: "{desc}". Hãy tạo một "Prompt Chuẩn" TRÌNH BÀY CHÍNH XÁC theo cấu trúc 4 mục sau:
                1. Tạo ứng dụng "[Tên ứng dụng]". Ứng dụng có chức năng "[Mô tả chức năng chính]".
                2. Đầu vào của ứng dụng là: [Liệt kê các đầu vào cần thiết].
                3. Đầu ra của ứng dụng là: [Mô tả kết quả mong muốn].
                4. Chức năng "[Tên chức năng chính]" được mô hình "{model_name_c}" xử lý. Kết nối với mô hình "{model_name_c}" thông qua giao thức sau: 
                curl --location 'https://api.cerebras.ai/v1/chat/completions' \\
                --header 'Content-Type: application/json' \\
                --header "Authorization: Bearer {current_key}" \\
                --data '{{
                  "model": "{model_id_c}",
                  "stream": true,
                  "max_tokens": {max_tokens_c},
                  "temperature": 1,
                  "top_p": 1,{reasoning_c}
                  "messages": [
                    {{
                      "role": "system",
                      "content": ""
                    }}
                  ]
                }}'
                
                Yêu cầu: Trả về văn bản thuần theo đúng cấu trúc trên, không thêm lời dẫn giải."""
                
            else:
                current_model_id = 'gemini-2.5-flash' if model_type == 'default' else 'gemini-3-flash-preview'
                model_display = 'Gemini 2.5 Flash' if model_type == 'default' else 'Gemini 3 Flash'

                full_prompt = f"""Dựa trên mô tả: "{desc}". Hãy tạo một "Prompt Chuẩn" TRÌNH BÀY CHÍNH XÁC theo cấu trúc 4 mục sau:
                1. Tạo ứng dụng "[Tên ứng dụng]". Ứng dụng có chức năng "[Mô tả chức năng chính]".
                2. Đầu vào của ứng dụng là: [Liệt kê các đầu vào cần thiết].
                3. Đầu ra của ứng dụng là: [Mô tả kết quả mong muốn].
                4. Chức năng "[Tên chức năng chính]" được mô hình "{model_display}" xử lý. Kết nối với mô hình "{model_display}" thông qua giao thức sau: https://generativelanguage.googleapis.com/v1beta/models/{current_model_id}:generateContent?key={current_key}
                
                Yêu cầu: Trả về văn bản thuần theo đúng cấu trúc trên, không thêm lời dẫn giải."""
                
        elif task_type == 'code':
            user_prompt = prompt
            if not user_prompt:
                return jsonify({'error': 'Prompt is required'}), 400
                
            system_instruction = (
                "You are an expert web developer. Use HTML, CSS, and JavaScript. "
                "Provide a complete, standalone HTML file. Do NOT use markdown code blocks. "
                "Just return the raw code."
            )
            
            task_instruction = f"Yêu cầu mới: {user_prompt}. Hãy cập nhật mã nguồn hoàn chỉnh dựa trên yêu cầu này." if context else f"Task: {user_prompt}"
            
            if context:
                full_prompt = f"{system_instruction}\n\nCode hiện tại:\n```html\n{context}\n```\n\n{task_instruction}"
            else:
                full_prompt = f"{system_instruction}\n\n{task_instruction}"
        else:
            return jsonify({'error': 'Invalid taskType'}), 400

        output_text = ""

        # 1. Gemini
        if model_type == 'default' or model_type == 'gemini-3-flash':
            token = api_key if api_key else DEFAULT_GEMINI_KEY
            model_id = 'gemini-2.5-flash' if model_type == 'default' else 'gemini-3-flash-preview'
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={token}"
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 65536
                }
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                output_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                return jsonify({'error': f"Gemini Error: {response.text}"}), response.status_code

        # 2. Groq
        elif 'groq' in model_type:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "qwen/qwen3-32b" if 'qwen' in model_type else "moonshotai/kimi-k2-instruct-0905",
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.6
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                output_text = response.json()['choices'][0]['message']['content']
            else:
                return jsonify({'error': f"Groq Error: {response.text}"}), response.status_code

        # 3. Cerebras
        elif 'cerebras' in model_type:
            if 'gpt-oss-120b' in model_type:
                model_id = 'gpt-oss-120b'
            elif 'llama3.1-8b' in model_type:
                model_id = 'llama3.1-8b'
            else:
                model_id = 'zai-glm-4.7' if 'glm' in model_type else 'qwen-3-235b-a22b-instruct-2507'
            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": full_prompt}]
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                output_text = response.json()['choices'][0]['message']['content']
            else:
                return jsonify({'error': f"Cerebras Error: {response.text}"}), response.status_code

        # 4. Huggingface
        elif 'huggingface' in model_type:
            token = api_key if api_key else DEFAULT_HF_KEY
            url = "https://router.huggingface.co/v1/chat/completions"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {
                "model": "moonshotai/Kimi-K2-Thinking:novita",
                "messages": [{"role": "user", "content": full_prompt}]
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                output_text = response.json()['choices'][0]['message']['content']
            else:
                return jsonify({'error': f"Huggingface Error: {response.text}"}), response.status_code

        # Post-processing
        clean_output = output_text.strip()
        
        # Specific for Llama3.1-8b (Cerebras) - Extract only HTML content
        if 'llama3.1-8b' in model_type and task_type == 'code':
            start_tag = "<!DOCTYPE html>"
            end_tag = "</html>"
            start_idx = clean_output.find(start_tag)
            end_idx = clean_output.rfind(end_tag)
            if start_idx != -1 and end_idx != -1:
                clean_output = clean_output[start_idx:end_idx + len(end_tag)].strip()

        if clean_output.startswith("```"):
            lines = clean_output.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            clean_output = "\n".join(lines).strip()
        
        if "</thinking>" in clean_output:
            clean_output = clean_output.split("</thinking>")[-1].strip()
        
        # Remove <think>...</think> tags (common in Qwen reasoning models)
        clean_output = re.sub(r'<think>.*?</think>', '', clean_output, flags=re.DOTALL).strip()

        return jsonify({'result': clean_output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
