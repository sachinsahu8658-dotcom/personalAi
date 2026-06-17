import os
import re
import math
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In-memory data store for the text you teach the engine
KNOWLEDGE_BASE = {
    "raw_text": "",
    "sentences": [],
    "word_frequencies": {},
    "timeline": []
}

STOPWORDS = {"the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of", "for", "by", "with", "was", "were", "from", "that", "as", "it", "its", "he", "she", "they", "this", "had", "been", "has", "have"}

def process_text(text):
    KNOWLEDGE_BASE["raw_text"] = text
    
    # Clean and split into individual sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    KNOWLEDGE_BASE["sentences"] = sentences

    # Compute local vocabulary weights (Term Frequency baseline)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    KNOWLEDGE_BASE["word_frequencies"] = freq

    # Build Chronological Timeline Matrix using Regex Year Extraction
    timeline = []
    for sentence in sentences:
        years = re.findall(r'\b(1[56789]\d{2}|20\d{2})\b', sentence)
        for year in years:
            timeline.append({"year": int(year), "event": sentence})
    
    # Sort timeline oldest to newest
    KNOWLEDGE_BASE["timeline"] = sorted(timeline, key=lambda x: x["year"])

@app.route('/teach', methods=['POST'])
def teach_engine():
    data = request.get_json()
    text = data.get('text', '').strip()
    if len(text) < 50:
        return jsonify({"error": "Please provide a more substantial lesson text (at least 50 characters)."})
    
    process_text(text)
    return jsonify({
        "success": True,
        "sentences_parsed": len(KNOWLEDGE_BASE["sentences"]),
        "events_mapped": len(KNOWLEDGE_BASE["timeline"])
    })

@app.route('/summary', methods=['GET'])
def get_summary():
    sentences = KNOWLEDGE_BASE["sentences"]
    freq = KNOWLEDGE_BASE["word_frequencies"]
    if not sentences:
        return jsonify({"summary": "No data trained yet."})

    # Score sentences based on the weight density of their unique keywords
    sentence_scores = []
    for s in sentences:
        s_words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        score = sum(freq.get(w, 0) for w in s_words if w not in STOPWORDS)
        # Normalize by length to avoid bias toward long sentences
        normalized_score = score / (len(s_words) + 1)
        sentence_scores.append((normalized_score, s))

    # Extract the top 3 highest-scoring structural sentences
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [item[1] for item in sentence_scores[:3]]
    
    # Re-order back to natural reading flow
    summary = " ".join([s for s in sentences if s in top_sentences])
    return jsonify({"summary": summary, "timeline": KNOWLEDGE_BASE["timeline"]})

@app.route('/quiz', methods=['GET'])
def generate_quiz():
    sentences = KNOWLEDGE_BASE["sentences"]
    if not sentences:
        return jsonify({"error": "Teach me a topic first before taking a quiz!"})

    # Find sentences containing factual targets (Years or key capitalized entities)
    quiz_pool = []
    for s in sentences:
        years = re.findall(r'\b(1[56789]\d{2}|20\d{2})\b', s)
        if years:
            quiz_pool.append(("year", years[0], s))
        else:
            capitals = re.findall(r'\b[A-Z][a-z]{3,}\b', s)
            # Avoid picking the first word of the sentence exclusively
            valid_caps = [c for c in capitals if not s.startswith(c)]
            if valid_caps:
                quiz_pool.append(("entity", valid_caps[0], s))

    if not quiz_pool:
        return jsonify({"error": "Could not extract clean testing metrics from this specific text segment."})

    # Pick a random factual sentence
    q_type, answer, full_sentence = random.choice(quiz_pool)
    question_text = full_sentence.replace(answer, "_______")

    # Generate Distractors (Options Pool)
    options = [answer]
    if q_type == "year":
        ans_int = int(answer)
        # Shift offsets to build mathematical distractors
        offsets = [-10, 5, 10, -5, 20]
        for offset in offsets:
            alt = str(ans_int + offset)
            if alt not in options:
                options.append(alt)
            if len(options) == 4: break
    else:
        # Pull alternative entities from the taught text pool
        all_words = re.findall(r'\b[A-Z][a-z]{3,}\b', KNOWLEDGE_BASE["raw_text"])
        alts = list(set([w for w in all_words if w != answer and not full_sentence.startswith(w)]))
        random.shuffle(alts)
        options.extend(alts[:3])

    # Fill if options fall short
    while len(options) < 4:
        options.append(f"Alternative Fact {len(options)}")

    random.shuffle(options)
    letter_map = ["A", "B", "C", "D"]
    correct_letter = letter_map[options.index(answer)]

    return jsonify({
        "question": f"Based on your notes: {question_text}",
        "options": [f"Option A: {options[0]}", f"Option B: {options[1]}", f"Option C: {options[2]}", f"Option D: {options[3]}"],
        "correct": correct_letter,
        "explanation": f"The exact source context reads: '{full_sentence}'"
    })

@app.route('/ask', methods=['POST'])
def query_knowledge():
    data = request.get_json()
    query = data.get('query', '').lower().strip()
    sentences = KNOWLEDGE_BASE["sentences"]
    
    if not sentences:
        return jsonify({"reply": "I have not been taught any material yet. Please paste your notes in the Teach tab."})

    query_words = [w for w in re.findall(r'\b[a-zA-Z]+\b', query) if w not in STOPWORDS]
    if not query_words:
        return jsonify({"reply": "Could you expand your question with more specific keywords?"})

    best_match = None
    highest_overlap = 0

    # Execute Vector-Style Overlap Metric Matching
    for s in sentences:
        s_words = set(re.findall(r'\b[a-zA-Z]+\b', s.lower()))
        overlap = sum(1 for w in query_words if w in s_words)
        
        if overlap > highest_overlap:
            highest_overlap = overlap
            best_match = s

    if highest_overlap > 0:
        return jsonify({"reply": best_match})
    return jsonify({"reply": "I couldn't locate an exact match for those criteria in the material you taught me. Try matching keywords from your text."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
        
