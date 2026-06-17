from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Import the neural network and the processor data
from brain import PersonalAI
from processor import VOCABULARY, text_to_vector, TRAINING_INPUTS, TRAINING_OUTPUTS

app = Flask(__name__)
CORS(app)

# --- IN-MEMORY HISTORY STORAGE ---
# This global list lives in the server's RAM until Render goes to sleep
CONVERSATION_HISTORY = []

# Initialize and train the AI Brain
ai = PersonalAI(input_size=len(VOCABULARY), hidden_size=8, output_size=3)
ai.train(TRAINING_INPUTS, TRAINING_OUTPUTS, epochs=2000, lr=0.1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_text = data.get("text", "").strip() 
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
        
    # Run the neural network prediction
    input_vector = text_to_vector(user_text)
    prediction = ai.forward(input_vector)[0]
    
    categories = ["General Knowledge (GK)", "Current Affairs (CA)", "English Grammar"]
    highest_match_idx = np.argmax(prediction)
    detected_category = categories[highest_match_idx]
    confidence_str = f"{float(prediction[highest_match_idx]) * 100:.1f}%"

    # --- SAVE TO LOCAL MEMORY LIST ---
    history_entry = {
        "query_text": user_text,
        "category": detected_category,
        "confidence": confidence_str
    }
    
    # Insert at the beginning (index 0) so the newest conversation shows up first
    CONVERSATION_HISTORY.insert(0, history_entry)
    
    # Keep only the last 20 conversations by trimming the list
    del CONVERSATION_HISTORY[20:]

    return jsonify({
        "query": user_text,
        "classification": detected_category,
        "confidence": confidence_str,
        "raw_probabilities": {
            "GK": f"{float(prediction[0])*100:.1f}%",
            "CA": f"{float(prediction[1])*100:.1f}%",
            "English": f"{float(prediction[2])*100:.1f}%"
        }
    })

# --- FETCH THE IN-MEMORY LOGS ---
@app.route('/history', methods=['GET'])
def get_history():
    # Simply send back the current state of our global Python list
    return jsonify(CONVERSATION_HISTORY)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
