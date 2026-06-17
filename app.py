from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Import the neural network and the processor data you just wrote
from brain import PersonalAI
from processor import VOCABULARY, text_to_vector, TRAINING_INPUTS, TRAINING_OUTPUTS

app = Flask(__name__)
CORS(app) # Allows your Spck Editor frontend to call this server safely

# Initialize the AI Brain
# Input size = length of our vocabulary (24 words)
# Hidden layer nodes = 8 (perfect for small, fast classification tasks)
# Output size = 3 (GK, Current Affairs, English)
ai = PersonalAI(input_size=len(VOCABULARY), hidden_size=8, output_size=3)

print("Training the Neural Network on baseline patterns...")
# Run an initial 2000 loops of training so the AI starts smart
ai.train(TRAINING_INPUTS, TRAINING_OUTPUTS, epochs=2000, lr=0.1)
print("Training complete! Brain is ready.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "AI is awake"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
        
    # 1. Convert sentence to numerical vector
    input_vector = text_to_vector(user_text)
    
    # 2. Run a forward pass through the network layers
    prediction = ai.forward(input_vector)[0]
    
    # 3. Extract probability confidence metrics
    categories = ["General Knowledge (GK)", "Current Affairs (CA)", "English Grammar"]
    highest_match_idx = np.argmax(prediction)
    detected_category = categories[highest_match_idx]
    confidence = float(prediction[highest_match_idx]) * 100

    return jsonify({
        "query": user_text,
        "classification": detected_category,
        "confidence": f"{confidence:.1f}%",
        "raw_probabilities": {
            "GK": f"{float(prediction[0])*100:.1f}%",
            "CA": f"{float(prediction[1])*100:.1f}%",
            "English": f"{float(prediction[2])*100:.1f}%"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
