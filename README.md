# 🧠 Al-medical-query-assistant 
### *(AI-Powered Disease Similarity & Recommendation Engine)*  

---

## 🚀 Overview
This project is an **AI-powered medical assistant** that analyzes user symptoms and provides possible **disease matches**, along with relevant **information, medicines, and precautions**.  

It uses **semantic search with embeddings** to identify diseases similar to a given symptom description and then leverages an **LLM (Llama 3 via Groq API)** to generate an intelligent response.

---

## 🩺 Features
✅ Converts a master disease dataset (`diseases_data.json`) into structured files for modular access.  
✅ Generates **semantic embeddings** for every disease using **Sentence Transformers**.  
✅ Computes similarity between user queries and stored embeddings to find **relevant diseases**.  
✅ Retrieves context and feeds it to a **Groq-hosted LLM** (`llama3-8b-8192`) for natural responses.  
✅ Outputs probable causes, medicines, and precautions in a human-like explanation format.  

---

## 🧩 Workflow

### 1. Data Preparation
- Reads `diseases_data.json` and creates separate `.json` files inside a `documents/` folder.  
- Each file contains structured information about a specific disease.

### 2. Embedding Generation
- Loads **`all-MiniLM-L6-v2`** SentenceTransformer.
- Encodes each disease file (`name + details`) into a vector embedding.
- Stores all embeddings in a list (`vectors`) for fast similarity computation.

### 3. Semantic Retrieval
- When a user enters a symptom-based query, the system:
  - Generates an embedding for the query.
  - Calculates **Euclidean distance** between query and disease embeddings.
  - Retrieves **top 4 most similar** disease files for context.

### 4. Intelligent Reasoning
- Constructs a detailed prompt containing:
  - User’s symptom-based query.
  - Retrieved context (top disease matches).  
- Sends this prompt to **Groq API** (Llama 3 model) to get an intelligent explanation.

### 5. Output
- Displays the generated answer suggesting:
  - Possible disease(s)
  - Medicines
  - Precautions or next steps  

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Language** | Python |
| **Embedding Model** | SentenceTransformer (`all-MiniLM-L6-v2`) |
| **Vector Handling** | NumPy, Pandas |
| **LLM API** | Groq (Llama 3-8B-8192) |
| **File Handling** | JSON, OS, Glob |
| **Similarity Metric** | Euclidean Distance |

---

### 🧩 File Structure
```
medical-symptom-analyzer/
│
├── detailed.ipynb               # detailed code explanation
├── diseases_data.json           # Master dataset (input)
├── documents/                   # Auto-generated disease files
│   ├── Fever.json
│   ├── Diabetes.json
│   └── ...
├── main.py                      # Core logic file
├── requirements.txt
└── README.md
```

---


## ⚙️ Installation & Setup

### 🔸 1. Clone the Repository
```bash
git clone https://github.com/yourusername/medical-symptom-analyzer.git
cd medical-symptom-analyzer
```

### 🔸 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 🔸 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Place your diseases_data.json in the project root.

### 2. Open main.py and replace the placeholder:
```bash
api_key = 'YOUR_API_KEY_HERE'
```
with your actual Groq API key.

### 3. Run the script:
```bash
python main.py
```

### 4. The system will:

    Generate the documents folder.
    
    Create embeddings.
    
    Retrieve relevant diseases.
    
    Query the Llama 3 model for an intelligent response.

--- 

### Example Query

input(sample)
```bash
question = "I am suffering with memory loss, what could be the reason?"
```
output(sample):
```bash
Possible cause: Alzheimer's Disease, Vitamin B12 deficiency, or depression.
Recommended actions: Consult a neurologist, perform cognitive tests, maintain healthy diet.
Medicines: Donepezil (as per doctor’s prescription)
Precautions: Regular brain exercises, avoid alcohol, ensure proper sleep.
```

---


## 🛠️ Functions Explained
| Function                         | Description                                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **get_embeddings_new(text)**     | Generates normalized embeddings for the given text using SentenceTransformer.                        |
| **get_similar_documents(query)** | Finds top-4 similar disease documents based on semantic similarity.                                  |
| **Groq API call**                | Sends the user query and contextual diseases to the Llama3 model and retrieves intelligent response. |

---
