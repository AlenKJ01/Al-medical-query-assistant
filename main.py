import os 
import json
import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
import openai 

with open('diseases_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

if not os.path.exists('documents'):
    os.makedirs('documents')

for disease, details in data.items():
    file_path = os.path.join('documents', f'{disease}.json')

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(details, file, indent=2, ensure_ascii=False)

print('Files created successfully in the documents folder')


model = SentenceTransformer('all-MiniLM-L6-v2')
print(model.max_seq_length)
model.max_seq_length = 512

def get_embeddings_new(text):
    embeddings = model.encode([text], normalize_embeddings=True)
    return embeddings

vectors = []

for i in glob.glob('documents/*'):
    with open(i, 'r', encoding='utf-8') as file:
        data = json.load(file)

        name = i.split('/')[-1].split('.')[0]

        tem_vector = get_embeddings_new(str(name) + ' ' + str(data))[0]

        tem = {}
        tem['name'] = name
        tem['vector'] = tem_vector
        vectors.append(tem)

def get_similar_documents(query):

    emb = get_embeddings_new(query)

    df = pd.DataFrame(vectors)
    df['distance'] = df['vector'].apply(lambda x: np.sum((x-emb)**2))
    files = list(i + '.json' for i in df.sort_values('distance')['name'].head(4))

    context = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as i:
            data = json.load(i)

            new_dict = {}
            new_dict['disease_name'] = file.split('.')[0]
            new_dict['disease_info'] = data
            #print(new_dict)
            context.append(new_dict)
    return str(context)

question = 'i am suffering with memory loss what could be the reason ?'
context = get_similar_documents(question)

payload = f'''

You are an AI assistant to help user to help with symptoms, possible diseases_name, medicines and precautions of a diseases 

to help with all those information please user our context provided below 

context = {context}

user_question = {question}

'''

client = openai.OpenAI(
    base_url = "https://api.groq.com/openai/v1", 
    api_key = 'YOUR_API_KEY_HERE' 
)

response = client.chat.completions.create(
    model = 'llama3-8b-8192',
    messages = [
        {
            'role' : 'user',
            'content' : [
                {
                    'type' : 'text',
                    'text' : payload
                }
            ]
        }
    ],
    temperature = 1,
    max_tokens= 2048,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
    response_format = {
        'type' : 'text'
    }
)

print(response.choices[0].message.content)