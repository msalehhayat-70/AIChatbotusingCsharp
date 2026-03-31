"""
TechBot - ML Inference Bridge
Called by C# via Process.Start to classify intent and return a response.
Usage: python predict.py "user input text here"
Output: JSON { "intent": "...", "response": "...", "confidence": 0.0 }
"""

import sys, os, json, pickle, random

def predict(user_input):
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "Model")

    with open(os.path.join(model_dir, "chatbot_model.pkl"), "rb") as f:
        bundle = pickle.load(f)

    with open(os.path.join(model_dir, "intent_responses.json")) as f:
        responses = json.load(f)

    from scipy.sparse import hstack

    text   = user_input.lower().strip()
    X_word = bundle["word_tfidf"].transform([text])
    X_char = bundle["char_tfidf"].transform([text])
    X      = hstack([X_word, X_char])

    clf    = bundle["clf"]
    intent = clf.predict(X)[0]

    # Confidence via softmax-like normalisation on decision scores
    scores    = clf.decision_function(X)[0]
    exp_s     = [2 ** float(s) for s in scores]
    total     = sum(exp_s)
    idx       = list(clf.classes_).index(intent)
    confidence = round(exp_s[idx] / total * 100, 1)

    # If confidence is very low, admit uncertainty
    if confidence < 8.0:
        response = (
            "I'm sorry, I don't have information about that. "
            "I can help with: Programming, AI, Cybersecurity, Networking, "
            "Hardware, Software, Web Development, Databases, "
            "Mobile Development, or Cloud Computing."
        )
        intent = "unknown"
    else:
        intent_responses = responses.get(intent, [
            "I'm sorry, I still have no idea about that. "
            "Try asking about programming, AI, cybersecurity, networking, "
            "hardware, software, web development, databases, "
            "mobile development, or cloud computing."
        ])
        response = random.choice(intent_responses)

    print(json.dumps({
        "intent":     intent,
        "response":   response,
        "confidence": confidence
    }))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"intent": "unknown", "response": "Please type a question!", "confidence": 0.0}))
    else:
        try:
            predict(" ".join(sys.argv[1:]))
        except Exception as e:
            print(json.dumps({"intent": "error", "response": f"Model error: {e}", "confidence": 0.0}))
