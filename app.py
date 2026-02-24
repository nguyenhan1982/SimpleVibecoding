import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Default keys (Fallbacks from config.js)
DEFAULT_GEMINI_KEY = "AIzaSyC4cZobo4bWS-g29Dtm8cGWp9aGQMBMR8E"
DEFAULT_HF_KEY = "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"

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt')
        model_type = data.get('model', 'default')
        api_key = data.get('apiKey')
        context = data.get('context', '')
        task_type = data.get('taskType', 'code') # code, ideas, prompt

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # System Instructions based on task type
        if task_type == 'code':
            system_instruction = (
                "You are an expert web developer. Use HTML, CSS, and JavaScript. "
                "Provide a complete, standalone HTML file. Do NOT use markdown code blocks. "
                "Just return the raw code."
            )
            if context:
                full_prompt = f"{system_instruction}\n\nExisting Code:\n{context}\n\nTask: {prompt}"
            else:
                full_prompt = f"{system_instruction}\n\nTask: {prompt}"
        else:
            full_prompt = prompt # Already formatted in JS for ideas/prompt

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
                "model": "moonshotai/kimi-k2-instruct-0905",
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
        if clean_output.startswith("```"):
            lines = clean_output.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            clean_output = "\n".join(lines).strip()
        
        if "</thinking>" in clean_output:
            clean_output = clean_output.split("</thinking>")[-1].strip()

        return jsonify({'result': clean_output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
