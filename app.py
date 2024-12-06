from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS
import torch
import re
import spacy


def load_saved_model(model_path):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to("cpu")
    model.eval()
    return model


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Replace these with your actual model and tokenizer
model_path = "E:/absa_farhan/aspect_based_sentiment_model.pth"
loaded_model = load_saved_model(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = LabelEncoder()
label_encoder.fit(["neutral", "negative", "positive"])


def split_text(text):

    split_by_newlines = [line.strip() for line in text.split("\n") if line.strip()]
    # split_by_newlines = text.split("\n")
    all_sentences = []
    for part in split_by_newlines:
        if part.strip():  # Ignore empty parts
            all_sentences.extend(split_on_conjunctions(part))
    return all_sentences


def split_on_conjunctions(text):

    doc = nlp(text)
    sentences = []
    current_sentence = []
    for token in doc:
        current_sentence.append(token.text)
        # If we encounter a coordinating conjunction or punctuation, split
        if token.dep_ in ["cc", "punct"] or token.text in [".", "!", "?"]:
            sentences.append(" ".join(current_sentence).strip())
            current_sentence = []
    if current_sentence:
        sentences.append(" ".join(current_sentence).strip())
    return sentences


def predict_sentiments(text, model, tokenizer, label_encoder, device):

    aspects = ["battery", "display", "design", "performance", "camera"]
    results = {}
    model.eval()

    for aspect in aspects:
        aspect_pattern = r"\b" + re.escape(aspect) + r"\b"
        if re.search(aspect_pattern, text, re.IGNORECASE):
            sentences = re.split(r"(?<=[.!?])\s+", text)
            relevant_sentence = next(
                (s for s in sentences if re.search(aspect_pattern, s, re.IGNORECASE)),
                text,
            )

            inputs = tokenizer.encode_plus(
                aspect,
                relevant_sentence,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(
                    inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    token_type_ids=inputs["token_type_ids"].to(device),
                )

            _, predicted = torch.max(outputs.logits, 1)

            try:
                sentiment = label_encoder.inverse_transform([int(predicted.item())])[0]
            except Exception as e:
                print(f"Error in label decoding: {e}")
                sentiment = "unknown"

            results[aspect] = sentiment

    return results


def predict_sentences(text, model, tokenizer, label_encoder, device="cpu"):
    """Predict sentiments for multiple sentences."""
    sentences = split_text(text)
    overall_results = {}

    for sentence in sentences:
        predictions = predict_sentiments(
            sentence, model, tokenizer, label_encoder, device
        )
        for aspect, sentiment in predictions.items():
            if aspect not in overall_results:
                overall_results[aspect] = []
            overall_results[aspect].append(sentiment)

    final_results = {}
    for aspect, sentiments in overall_results.items():
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        final_results[aspect] = final_sentiment

    return final_results


# Define API routes
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    if "text" not in data:
        return jsonify({"error": "Invalid input. Text is required."}), 400

    text = data["text"]

    try:
        predictions = predict_sentences(
            text, loaded_model, tokenizer, label_encoder, device="cpu"
        )
        return jsonify({"results": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
