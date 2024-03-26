import openai
import os
import csv
import glob
from dotenv import load_dotenv
from itertools import zip_longest

load_dotenv()

# Function to split text into chunks
def chunk_text(text, max_chunk_length=8000):
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    return chunks

text_array = []
api_key = os.environ.get('OPENAI_KEY')
openai.api_key = api_key
dir_path = os.path.join(os.getcwd(), 'documents1')
dir_full_path = os.path.join(dir_path, '*.txt')
embeddings_filename = "embeddings.csv"

# This array is used to store the embeddings
embedding_array = []

if api_key is None or api_key == "YOUR_OPENAI_KEY_HERE":
    print("Invalid API key")
    exit()

# Loop through all .txt files in the /training-data folder
for file in glob.glob(dir_full_path):
    # Read the data from each file and push to the array
    # The dump method is used to convert spacings into newline characters \n
    with open(file, 'r') as f:
        text = f.read().replace('\n', '')
        text_array.append(text)

# Define the batch size
batch_size = 5

# Process texts in batches
for i in range(0, len(text_array), batch_size):
    batch_texts = text_array[i:i+batch_size]
    processed_batch_texts = []

    # Split texts into smaller chunks if necessary
    for text in batch_texts:
        if len(text) > 8192:
            chunks = chunk_text(text)
            processed_batch_texts.extend(chunks)
        else:
            processed_batch_texts.append(text)

    embeddings = openai.Embedding.create(
        input=processed_batch_texts,
        model="text-embedding-3-large"
    )['data']

    for response, text in zip_longest(embeddings, batch_texts):
        if response is not None:
            embedding_dict = {'embedding': response['embedding'], 'text': text}
            embedding_array.append(embedding_dict)

with open(embeddings_filename, 'w', newline='') as f:
    # This sets the headers
    fieldnames = ['embedding', 'text']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for obj in embedding_array:
        # The embedding vector will be stored as a string to avoid comma
        # separated issues between the values in the CSV
        writer.writerow({'embedding': str(obj['embedding']), 'text': obj['text']})

print("Embeddings saved to:", embeddings_filename)
