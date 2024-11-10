from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from flask import jsonify, Flask, request

app = Flask(__name__)

checkpoint = "Salesforce/codet5p-220m-bimodal"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
device = "cuda"  # for GPU usage or "cpu" for CPU usage

def sliding_window_tokenize(text, tokenizer, max_length=512, stride=50):
    tokenized_text = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length"  # Ensures each window has the same length
    )
    return tokenized_text['input_ids']

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    code = data.get('code', '')
    if not code:
        return jsonify({"error": "No code provided"}), 400

    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    prompt = (
        f"Given the following code, please analyze and answer the following:\n\n"
        f"1. What is the main functionality of this code?\n"
        f"2. Are there any potential performance improvements?\n"
        f"3. Are there any readability improvements you would suggest?\n"
        f"\nCode:\n{code}"
    )
    input_ids = sliding_window_tokenize(prompt, tokenizer).to(device)

    generated_ids = model.generate(input_ids, max_length=1000)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return jsonify({"insight": result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)