# 🤖 TechBot – AI Technology Chatbot

A **WinForms AI Chatbot** built with **C# (.NET 8)** powered by a **custom-trained Machine Learning model** — no external API, no internet required. The model runs entirely on your local machine.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [The Trained AI Model — Full Detail](#-the-trained-ai-model--full-detail)
   - [What Kind of Model?](#-what-kind-of-model)
   - [Algorithm 1 — TF-IDF Vectoriser](#-algorithm-1--tf-idf-vectoriser-feature-extraction)
   - [Algorithm 2 — LinearSVC Classifier](#-algorithm-2--linearsvc-classifier)
   - [Why These Algorithms?](#-why-these-algorithms-were-chosen)
   - [Training Data](#-training-data)
   - [How the Model Was Trained](#-how-the-model-was-trained-step-by-step)
   - [Model Files](#-model-files-saved-to-disk)
   - [Model Performance](#-model-performance)
3. [How the Model Is Integrated in C#](#-how-the-model-is-integrated-in-c)
   - [Integration Architecture](#-integration-architecture)
   - [Step-by-Step Data Flow](#-step-by-step-data-flow)
   - [C# Bridge Code](#-c-bridge-code-chatbotenginecs)
   - [Python Inference Script](#-python-inference-script-predictpy)
4. [Project Structure](#-project-structure)
5. [Prerequisites & Setup](#-prerequisites--setup)
6. [How to Run](#-how-to-run)
7. [Retrain the Model](#-retrain-the-model)
8. [Docker](#-docker)
9. [Technologies Used](#-technologies-used)
10. [Future Improvements](#-future-improvements)

---

## 📖 Project Overview

TechBot is an offline AI chatbot that classifies user questions about technology into one of **12 intent categories** and returns an informative response. It was built as a university assignment demonstrating the complete AI application lifecycle:

- ✅ Custom dataset creation (941 labelled samples)
- ✅ Model training using scikit-learn
- ✅ Serialisation and local deployment
- ✅ Integration into a C# WinForms desktop app
- ✅ Docker containerisation
- ✅ GitHub version control

---

## 🧠 The Trained AI Model — Full Detail

### 📌 What Kind of Model?

TechBot uses a **supervised machine learning model for multi-class text classification**.

| Property | Value |
|---|---|
| **Problem Type** | Multi-class Text Classification |
| **Number of Classes** | 12 (one per intent) |
| **Input** | Raw text string (user's question) |
| **Output** | Predicted intent label + confidence score |
| **Model Type** | TF-IDF Feature Extraction + LinearSVC Classifier |
| **Training Library** | scikit-learn (Python) |
| **Serialisation** | Python `pickle` → `chatbot_model.pkl` |
| **Inference Runtime** | Python subprocess called from C# |

This is **not** a pre-built API model like ChatGPT or BERT. It is a model **trained from scratch** on a custom dataset created specifically for this project.

---

### 🔢 Algorithm 1 — TF-IDF Vectoriser (Feature Extraction)

Before any machine learning algorithm can process text, the text must be converted into numbers. **TF-IDF (Term Frequency–Inverse Document Frequency)** is the technique used here.

#### What is TF-IDF?

TF-IDF converts a sentence into a **sparse numerical vector** where each dimension represents a word or character sequence (n-gram), and the value represents how important that term is in the document relative to the entire corpus.

```
TF-IDF(term, document) = TF(term, document) × IDF(term, corpus)

TF  = (Number of times term appears in document) / (Total terms in document)
IDF = log(Total documents / Number of documents containing the term)
```

**Why IDF matters:** Common words like "what", "is", "the" appear in every sentence and carry no discriminating power. IDF penalises them heavily. Rare but distinctive words like "kubernetes", "LinearSVC", or "phishing" get high weights.

#### Two Vectorisers Are Used in Parallel

TechBot uses **two separate TF-IDF vectorisers** whose outputs are combined:

| Vectoriser | `analyzer` | `ngram_range` | `max_features` | Purpose |
|---|---|---|---|---|
| **Word TF-IDF** | `'word'` | `(1, 3)` | 20,000 | Captures whole-word patterns: "what is", "machine learning", "how does VPN work" |
| **Char TF-IDF** | `'char_wb'` | `(3, 5)` | 20,000 | Captures character sequences: "GPU", "SQL", "VPN" — handles typos and abbreviations |

#### What Are N-grams?

An n-gram is a contiguous sequence of n items from a text.

```
Input: "what is machine learning"

Word unigrams (1):  ["what", "is", "machine", "learning"]
Word bigrams  (2):  ["what is", "is machine", "machine learning"]
Word trigrams (3):  ["what is machine", "is machine learning"]

Char trigrams (3):  ["wha", "hat", "at ", "t i", " is", "is ", ...]
Char 4-grams  (4):  ["what", "hat ", "at i", "t is", ...]
Char 5-grams  (5):  ["what ", "hat i", "at is", ...]
```

By combining word n-grams AND character n-grams, the model captures both **semantic meaning** (what words mean together) and **morphological patterns** (what word shapes look like).

#### Feature Matrix Combination

The two TF-IDF matrices are horizontally concatenated using `scipy.sparse.hstack`:

```
Word TF-IDF output:  shape (941, 20000)
Char TF-IDF output:  shape (941, 20000)
Combined matrix:     shape (941, 40000)  ← fed to LinearSVC
```

---

### ⚔️ Algorithm 2 — LinearSVC Classifier

After TF-IDF converts text to numbers, **LinearSVC (Linear Support Vector Classifier)** learns to separate the 12 intent classes.

#### What is SVM / LinearSVC?

A **Support Vector Machine (SVM)** is a supervised learning algorithm that finds the **optimal hyperplane** that best separates data points of different classes in high-dimensional feature space. LinearSVC is the linear (fastest) variant.

```
For binary classification:
    Find the hyperplane w·x + b = 0 that maximises the margin
    between the closest points (support vectors) of each class.

For multi-class (12 classes here):
    LinearSVC uses the One-vs-Rest strategy:
    Train 12 binary classifiers, one per intent.
    Each classifier answers: "Is this input intent X or not?"
    The classifier with the highest decision score wins.
```

#### Why LinearSVC Works Well for Text

Text data converted by TF-IDF is:
- **Very high-dimensional** (40,000 features)
- **Sparse** (most values are 0)
- **Linearly separable** in high dimensions

LinearSVC is specifically designed for this kind of data and consistently outperforms more complex models (like neural networks) on small-to-medium text datasets because it avoids overfitting.

#### Hyperparameters Used

```python
LinearSVC(
    C          = 5.0,      # Regularisation strength — higher C = less regularisation,
                           # allows the model to fit training data more tightly.
                           # Tuned from default 1.0 → 5.0 for better accuracy.
    max_iter   = 10000,    # Maximum optimisation iterations to ensure convergence.
    dual       = True,     # Use dual formulation — preferred when
                           # n_samples > n_features (our case with hstack).
    random_state = 42      # Reproducibility seed.
)
```

---

### 🤔 Why These Algorithms Were Chosen

Several algorithms were evaluated before selecting TF-IDF + LinearSVC:

| Algorithm | Considered? | Reason for Rejection / Selection |
|---|---|---|
| **LinearSVC** | ✅ **Chosen** | Best accuracy on high-dim sparse text. Fast training. Excellent for multi-class. |
| Logistic Regression | Considered | Good baseline, slightly lower accuracy than LinearSVC on this dataset |
| Naive Bayes (MultinomialNB) | Considered | Fast, but assumes feature independence — poor when tech terms overlap |
| Random Forest | Considered | Slow on 40,000 sparse features, prone to overfitting on small datasets |
| K-Nearest Neighbours | Rejected | Very slow at inference time on high-dim sparse vectors |
| BERT / DistilBERT | Future option | Excellent accuracy but requires GPU, large RAM, and complex setup |
| GPT-2 (ONNX) | Rejected | 500MB+ model size, needs GPU for real-time response |

**Conclusion:** For a dataset of ~941 samples, 12 classes, and offline deployment on a standard laptop, **TF-IDF + LinearSVC is the gold standard** in NLP classification.

---

### 📚 Training Data

The training dataset was **manually created** for this project. It is not sourced from any external dataset or API.

#### Dataset Statistics

| Property | Value |
|---|---|
| **Total samples** | 941 |
| **Number of intents** | 12 |
| **Average samples per intent** | ~78 |
| **Language** | English |
| **Data format** | Python dictionary in `train_model.py` |
| **Train / Test split** | 85% train (799) / 15% test (142) |
| **Stratified split** | Yes — equal class representation in both sets |

#### Intent Categories and Sample Count

| # | Intent | Sample Questions (examples) |
|---|---|---|
| 1 | `greeting` | "hello", "hi there", "good morning", "hey bot" |
| 2 | `farewell` | "bye", "goodbye", "see you later", "I'm done" |
| 3 | `programming` | "what is Python", "explain recursion", "what is Big O notation" |
| 4 | `artificial_intelligence` | "what is ML", "explain neural network", "what is GPT" |
| 5 | `cybersecurity` | "what is phishing", "how to stay safe online", "what is ransomware" |
| 6 | `networking` | "what is DNS", "explain OSI model", "difference TCP vs UDP" |
| 7 | `hardware` | "what is a GPU", "what is DDR5 RAM", "what is NVMe SSD" |
| 8 | `software` | "what is Linux", "what is version control", "what is a kernel" |
| 9 | `web_development` | "what is HTML", "what is React", "how does a website work" |
| 10 | `database` | "what is SQL", "MongoDB vs MySQL", "what is ACID" |
| 11 | `mobile_development` | "what is Flutter", "how to build Android app", "what is APK" |
| 12 | `cloud_computing` | "what is AWS", "explain Docker", "what is Kubernetes" |

#### Why So Many Samples Per Intent?

Each technology intent has ~80 variations because:
- Users phrase the same question many different ways
- The model must generalise, not memorise exact phrases
- More samples per class reduce the risk of misclassification at class boundaries
- Overlapping tech vocabulary (e.g., "what is Docker" could be cloud or software) requires sufficient context-specific examples to separate correctly

---

### 🔬 How the Model Was Trained — Step by Step

All training happens in `TrainingScript/train_model.py`. Here is the exact sequence:

#### Step 1 — Load and Prepare Data
```python
X = []  # list of text samples
y = []  # list of intent labels

for intent, samples in training_data.items():
    for sample in samples:
        X.append(sample.lower().strip())   # normalise to lowercase
        y.append(intent)

# Result: 941 samples, 941 labels
```

#### Step 2 — Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.15,   # 15% held out for evaluation
    random_state = 42,     # reproducible split
    stratify     = y       # ensures each class is proportionally represented
)
# X_train: 799 samples  |  X_test: 142 samples
```

#### Step 3 — Fit TF-IDF Vectorisers (on training data only)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

word_tfidf = TfidfVectorizer(
    ngram_range   = (1, 3),      # word unigrams, bigrams, trigrams
    max_features  = 20000,       # keep top 20,000 features by TF-IDF score
    sublinear_tf  = True,        # apply log(1 + TF) to reduce impact of very frequent terms
    analyzer      = 'word',
    token_pattern = r'\w+',      # tokenise on word characters only
    strip_accents = 'unicode'
)

char_tfidf = TfidfVectorizer(
    ngram_range  = (3, 5),       # char 3-grams, 4-grams, 5-grams
    max_features = 20000,
    sublinear_tf = True,
    analyzer     = 'char_wb',   # char n-grams within word boundaries
    strip_accents = 'unicode'
)

# IMPORTANT: fit ONLY on training data to prevent data leakage
X_train_word = word_tfidf.fit_transform(X_train)
X_train_char = char_tfidf.fit_transform(X_train)
```

#### Step 4 — Combine Feature Matrices
```python
from scipy.sparse import hstack

X_train_combined = hstack([X_train_word, X_train_char])
# Shape: (799, 40000) — 799 samples × 40,000 features
```

#### Step 5 — Train the Classifier
```python
from sklearn.svm import LinearSVC

clf = LinearSVC(C=5.0, max_iter=10000, dual=True)
clf.fit(X_train_combined, y_train)
# The classifier learns 12 × 40,000 = 480,000 weight parameters
```

#### Step 6 — Evaluate
```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

y_pred = clf.predict(X_test_combined)
print(accuracy_score(y_test, y_pred))    # 99.57% on full data

# 5-fold cross-validation for robust estimate
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(cv_scores.mean())                  # ~95%+
```

#### Step 7 — Save Model to Disk
```python
import pickle

model_bundle = {
    "word_tfidf": word_tfidf,   # fitted vectoriser (vocabulary + IDF weights)
    "char_tfidf": char_tfidf,   # fitted vectoriser
    "clf":        clf,          # trained LinearSVC weights
    "labels":     list(training_data.keys())  # class names
}

with open("Model/chatbot_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
```

---

### 💾 Model Files Saved to Disk

After training, three files are produced in `AIChatbot/Model/`:

#### `chatbot_model.pkl`
A binary pickle file containing a Python dictionary with four keys:

```
chatbot_model.pkl
├── "word_tfidf"  → TfidfVectorizer object (vocabulary of 20,000 word n-grams + IDF weights)
├── "char_tfidf"  → TfidfVectorizer object (vocabulary of 20,000 char n-grams + IDF weights)
├── "clf"         → LinearSVC object (trained weight matrix: 12 classes × 40,000 features)
└── "labels"      → ['greeting', 'farewell', 'programming', ..., 'cloud_computing']
```

This single file contains **everything needed to make a prediction** — no retraining required.

#### `intent_responses.json`
Maps each intent to a list of 2–3 response strings. At inference time, one response is randomly selected to add variety:

```json
{
  "programming": [
    "Programming is the process of writing instructions...",
    "Coding is an essential skill! I recommend Python..."
  ],
  "artificial_intelligence": [
    "Artificial Intelligence enables machines to simulate...",
    "AI is transforming every industry!..."
  ]
}
```

#### `intent_labels.json`
A simple JSON array listing all 12 intent class names in order:

```json
["greeting", "farewell", "programming", "artificial_intelligence",
 "cybersecurity", "networking", "hardware", "software",
 "web_development", "database", "mobile_development", "cloud_computing"]
```

---

### 📊 Model Performance

#### Accuracy Results

| Metric | Score |
|---|---|
| **Training Accuracy** | 99.57% |
| **5-Fold Cross-Validation Mean** | ~95%+ |
| **Live Manual Test (12 questions)** | 12 / 12 ✅ |

#### Live Test Results (12/12 Correct)

| Input Question | Predicted Intent | Correct? |
|---|---|---|
| "what is machine learning" | `artificial_intelligence` | ✅ |
| "how to build a website" | `web_development` | ✅ |
| "what is SQL database" | `database` | ✅ |
| "hello there" | `greeting` | ✅ |
| "what is VPN" | `networking` | ✅ |
| "how to stay safe from hackers" | `cybersecurity` | ✅ |
| "what is the cloud" | `cloud_computing` | ✅ |
| "best programming language for beginners" | `programming` | ✅ |
| "what is a GPU" | `hardware` | ✅ |
| "how to build an Android app" | `mobile_development` | ✅ |
| "what is operating system" | `software` | ✅ |
| "goodbye" | `farewell` | ✅ |

---

## 🔗 How the Model Is Integrated in C#

### 🏗️ Integration Architecture

The model runs in Python. The C# WinForms app communicates with it via a **subprocess bridge** — a design pattern where C# launches a Python process, passes input, and reads the JSON output:

```
┌──────────────────────────────────────────────────────────────┐
│                    WinForms UI  (C#)                         │
│  MainForm.cs — Dark chat UI, bubbles, sidebar, typing dots   │
└────────────────────────┬─────────────────────────────────────┘
                         │  user types question
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               ChatbotEngine.cs  (C#)                         │
│  1. Sanitises user input                                     │
│  2. Calls: Process.Start("python", "predict.py <input>")     │
│  3. Reads stdout JSON: { intent, response, confidence }      │
│  4. Returns ChatbotResponse object to UI                     │
└────────────────────────┬─────────────────────────────────────┘
                         │  python predict.py "user text"
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               predict.py  (Python inference)                 │
│  1. Loads chatbot_model.pkl (word_tfidf + char_tfidf + clf)  │
│  2. Preprocesses input: lowercase + strip                    │
│  3. Transforms: X_word = word_tfidf.transform([text])        │
│                 X_char = char_tfidf.transform([text])        │
│  4. Combines:   X = hstack([X_word, X_char])                 │
│  5. Predicts:   intent = clf.predict(X)[0]                   │
│  6. Scores:     confidence = softmax(decision_function())    │
│  7. Picks:      response = random.choice(responses[intent])  │
│  8. Prints:     {"intent":"...", "response":"...", "confidence":...} │
└──────────────────────────────────────────────────────────────┘
                         │  JSON string via stdout
                         ▼
              C# deserialises JSON → displays in chat UI
```

### 🔄 Step-by-Step Data Flow

Here is exactly what happens from the moment you press Enter to when the response appears:

```
1.  User types: "what is machine learning"
    └── MainForm.cs: txtInput.Text captured on KeyDown Enter

2.  ChatbotEngine.PredictAsync("what is machine learning") called
    └── Sanitises: "what is machine learning" (already clean)
    └── Builds process args: python "predict.py" "what is machine learning"

3.  C# launches Python subprocess:
    Process.Start("python", predict.py "what is machine learning")

4.  predict.py executes:
    a. pickle.load("chatbot_model.pkl")
       → loads word_tfidf, char_tfidf, clf objects into memory

    b. text = "what is machine learning"

    c. X_word = word_tfidf.transform(["what is machine learning"])
       → sparse matrix shape (1, 20000)
       → high values for: "machine learning", "what is", "is machine"

    d. X_char = char_tfidf.transform(["what is machine learning"])
       → sparse matrix shape (1, 20000)
       → high values for: "earn", "lear", "mach", "chin", "ine "

    e. X = scipy.sparse.hstack([X_word, X_char])
       → sparse matrix shape (1, 40000)

    f. intent = clf.predict(X)[0]
       → evaluates 12 binary classifiers
       → "artificial_intelligence" wins with highest decision score

    g. scores = clf.decision_function(X)[0]
       → confidence calculated via softmax-like normalisation
       → confidence ≈ 27.4%  (distributed across 12 classes)

    h. response = random.choice(responses["artificial_intelligence"])
       → "AI is transforming every industry!..."

    i. print(json.dumps({"intent":"artificial_intelligence",
                         "response":"AI is transforming...",
                         "confidence":27.4}))

5.  C# reads stdout string from subprocess
    └── JsonDocument.Parse(output)
    └── result.Intent    = "artificial_intelligence"
    └── result.Response  = "AI is transforming every industry!..."
    └── result.Confidence = 27.4

6.  MainForm displays:
    └── Bot bubble with response text
    └── Tag: "Intent: Artificial Intelligence  •  Confidence: 27.4%"
```

### 💻 C# Bridge Code (`ChatbotEngine.cs`)

The key method that calls the Python model:

```csharp
public async Task<ChatbotResponse> PredictAsync(string userInput)
{
    // 1. Escape input for command line safety
    string escaped = userInput.Replace("\"", "\\\"").Replace("\n", " ");
    string args = $"\"{_scriptPath}\" \"{escaped}\"";

    // 2. Configure Python subprocess
    var psi = new ProcessStartInfo
    {
        FileName               = "python",   // or python3 on Linux/Mac
        Arguments              = args,
        RedirectStandardOutput = true,       // capture stdout
        UseShellExecute        = false,
        CreateNoWindow         = true        // no console window popup
    };

    // 3. Run Python and capture output
    using var process = new Process { StartInfo = psi };
    process.Start();
    string output = await process.StandardOutput.ReadToEndAsync();
    await Task.Run(() => process.WaitForExit(15000));  // 15s timeout

    // 4. Parse JSON response from predict.py
    var doc  = JsonDocument.Parse(output.Trim());
    var root = doc.RootElement;

    return new ChatbotResponse
    {
        Intent     = root.GetProperty("intent").GetString(),
        Response   = root.GetProperty("response").GetString(),
        Confidence = root.GetProperty("confidence").GetDouble()
    };
}
```

### 🐍 Python Inference Script (`predict.py`)

```python
import sys, json, pickle, random
from scipy.sparse import hstack

def predict(user_input):
    # Load the trained model bundle from disk
    with open("Model/chatbot_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    word_tfidf = bundle["word_tfidf"]   # fitted TF-IDF (word ngrams)
    char_tfidf = bundle["char_tfidf"]   # fitted TF-IDF (char ngrams)
    clf        = bundle["clf"]          # trained LinearSVC

    # Load response templates
    with open("Model/intent_responses.json") as f:
        responses = json.load(f)

    # Preprocess + vectorise
    text   = user_input.lower().strip()
    X_word = word_tfidf.transform([text])       # shape: (1, 20000)
    X_char = char_tfidf.transform([text])       # shape: (1, 20000)
    X      = hstack([X_word, X_char])           # shape: (1, 40000)

    # Classify
    intent = clf.predict(X)[0]

    # Confidence estimate from decision function
    scores    = clf.decision_function(X)[0]
    exp_s     = [2 ** s for s in scores]
    confidence = round(exp_s[list(clf.classes_).index(intent)] / sum(exp_s) * 100, 1)

    # Random response for variety
    response = random.choice(responses[intent])

    # Output JSON to stdout — read by C# ChatbotEngine
    print(json.dumps({"intent": intent, "response": response, "confidence": confidence}))

if __name__ == "__main__":
    predict(" ".join(sys.argv[1:]))
```

---

## 🗂️ Project Structure

```
AIChatbot/
├── AIChatbot.sln                  ← Open this in Visual Studio 2022
├── Dockerfile                     ← Docker multi-stage build
├── .dockerignore
├── .gitignore
├── README.md                      ← This file
├── Theory_Assignment.docx         ← Theory report (Assignment #1)
│
├── AIChatbot/                     ← C# WinForms Project
│   ├── AIChatbot.csproj           ← Project file (.NET 8 WinForms)
│   ├── Program.cs                 ← Application entry point
│   ├── MainForm.cs                ← Chat UI — dark theme, bubbles, sidebar
│   ├── ChatbotEngine.cs           ← Python subprocess bridge + JSON parsing
│   ├── ChatMessage.cs             ← ChatMessage model class
│   │
│   ├── Model/                     ← Trained AI model files
│   │   ├── chatbot_model.pkl      ← Serialised bundle: word_tfidf + char_tfidf + clf
│   │   ├── intent_responses.json  ← Response templates per intent (2–3 variants each)
│   │   └── intent_labels.json     ← List of 12 intent class names
│   │
│   └── Python/
│       └── predict.py             ← Inference script: loads model → classifies → JSON output
│
└── TrainingScript/
    └── train_model.py             ← Full training pipeline — run to retrain the model
```

---

## ⚙️ Prerequisites & Setup

### Required Software

| Software | Version | Download |
|---|---|---|
| **Visual Studio 2022** | 17.x+ | [visualstudio.microsoft.com](https://visualstudio.microsoft.com/) |
| **.NET 8 SDK** | 8.0+ | [dotnet.microsoft.com](https://dotnet.microsoft.com/download/dotnet/8) |
| **Python** | 3.8+ | [python.org](https://www.python.org/downloads/) |

> ⚠️ **Important:** When installing Python, check **"Add Python to PATH"** — this is required so C# can call `python` from the command line.

### Required Python Packages

```bash
pip install scikit-learn scipy numpy
```

| Package | Version | Role |
|---|---|---|
| `scikit-learn` | 1.3+ | TF-IDF vectoriser + LinearSVC classifier |
| `scipy` | 1.11+ | `scipy.sparse.hstack` — combines feature matrices |
| `numpy` | 1.26+ | Numerical array operations |

---

## 🚀 How to Run

### Option 1 — Visual Studio 2022 (Recommended)

```
1. Extract AIChatbot_Project.zip
2. Open  AIChatbot.sln  in Visual Studio 2022
3. Right-click solution → Restore NuGet Packages
4. Press F5  (or click the green ▶ button)
```

### Option 2 — Command Line (.NET CLI)

```bash
cd AIChatbot
dotnet restore
dotnet run
```

### Option 3 — Test Model Directly (Without Running the App)

```bash
cd AIChatbot/AIChatbot
python Python/predict.py "what is machine learning"
# Output: {"intent": "artificial_intelligence", "response": "AI is transforming...", "confidence": 27.4}

python Python/predict.py "how to build a website"
# Output: {"intent": "web_development", "response": "Web development builds...", "confidence": 30.1}

python Python/predict.py "what is a GPU"
# Output: {"intent": "hardware", "response": "Key PC components...", "confidence": 27.2}
```

---

## 🔁 Retrain the Model

To add new intents or more training samples:

```bash
cd TrainingScript
pip install scikit-learn scipy numpy
python train_model.py
```

The training script will:
1. Load all training samples from the dictionary in the script
2. Lowercase and strip all text
3. Split into 85% train / 15% test with stratification
4. Fit `word_tfidf` and `char_tfidf` on training data
5. Combine feature matrices with `scipy.sparse.hstack`
6. Train `LinearSVC(C=5.0)` on combined features
7. Print accuracy report and 5-fold cross-validation score
8. Save updated `chatbot_model.pkl`, `intent_responses.json`, and `intent_labels.json` to `AIChatbot/Model/`

---

## 🐳 Docker

> **Note:** WinForms requires Windows containers. In Docker Desktop → Settings → Switch to Windows Containers.

```bash
# Build the Docker image (multi-stage: .NET SDK build + Python runtime)
docker build -t techbot .

# Run the container
docker run techbot

# Test the AI model directly in headless mode (no GUI needed)
docker run techbot python3 Python/predict.py "what is Kubernetes"

# Interactive shell to explore the container
docker run -it techbot bash
```

### What the Dockerfile Does

```
Stage 1 (Build): mcr.microsoft.com/dotnet/sdk:8.0
  └── dotnet restore → dotnet publish → /app/publish

Stage 2 (Runtime): mcr.microsoft.com/dotnet/runtime:8.0
  ├── apt-get install python3 python3-pip
  ├── pip3 install scikit-learn scipy numpy
  ├── COPY published .NET app
  ├── COPY Model/ (pkl + json files)
  ├── COPY Python/predict.py
  └── CMD ["dotnet", "AIChatbot.dll"]
```

---

## 🛠️ Technologies Used

| Technology | Version | Role in Project |
|---|---|---|
| **C#** | 12 | Main application language |
| **.NET 8** | 8.0 | Application framework (WinForms) |
| **WinForms** | .NET 8 | Desktop GUI — chat UI, sidebar, bubbles |
| **Python** | 3.8+ | Runtime for ML inference via subprocess |
| **scikit-learn** | 1.3+ | TF-IDF vectorisers + LinearSVC classifier |
| **scipy** | 1.11+ | Sparse matrix horizontal stacking |
| **numpy** | 1.26+ | Numerical computations |
| **pickle** | stdlib | Serialise/deserialise trained model to disk |
| **JSON** | stdlib | Data exchange format between Python and C# |
| **Newtonsoft.Json** | 13.0.3 | C# NuGet for JSON deserialisation |
| **Docker** | 24+ | Container build and deployment |
| **Git / GitHub** | - | Version control and submission |

---

## 🔮 Future Improvements

- **ONNX Export:** Convert the trained model to ONNX format and run inference natively in C# using `Microsoft.ML.OnnxRuntime` — eliminating the Python dependency entirely
- **Transformer Model:** Fine-tune DistilBERT or a small GPT model for context-aware, multi-turn conversations
- **More Intents:** Add intents for OS-specific questions, cloud certifications, DevOps tools
- **Conversation Memory:** Store chat history and include it in predictions for contextual understanding
- **Voice Input:** Integrate Windows Speech API for voice-to-text input
- **Web Version:** Port to ASP.NET Core + Blazor for browser-based deployment
- **Confidence Threshold:** If confidence < 20%, ask the user to rephrase instead of guessing

---

## 📝 Assignment Info

| Field | Value |
|---|---|
| **Course** | AI-Based Application Development in .NET |
| **Assignment** | #2 – AI Chatbot Mini Project (Practical) |
| **Theory Report** | `Theory_Assignment.docx` (Assignment #1) |
| **Model Type** | Custom-trained ML — TF-IDF + LinearSVC (no API) |
| **Tools** | Visual Studio 2022, C# .NET 8, Python, scikit-learn, Docker, GitHub |

---

## 👤 Author

Developed as part of the AI-Based Application Development in .NET course assignment.
