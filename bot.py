import json
import openai
import csv
import os
from dotenv import load_dotenv

load_dotenv()

# File location
embeddings_filename = "embeddings.csv"
company_name = "KissFlow"

openai.api_key = os.environ.get('OPENAI_KEY')
# This function remains unchanged
def calculate_similarity(vec1, vec2):
    dot_product = sum([vec1[i] * vec2[i] for i in range(len(vec1))])
    magnitude1 = sum([vec1[i] ** 2 for i in range(len(vec1))]) ** 0.5
    magnitude2 = sum([vec2[i] ** 2 for i in range(len(vec2))]) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


# New function to load embeddings and texts
def load_embeddings_and_texts(filename):
    data = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assuming the embedding column is named 'embedding',
            # and there is a column named 'text' for the original text
            embedding = json.loads(row['embedding'])
            text = row['text']
            data.append({'embedding': embedding, 'text': text})
    return data


# Modified chat function to pre-load data
def chat():
    print(f"Welcome to {company_name} Knowledge Base. How can I help you?")
    print("Type 'quit' to exit.")

    # Pre-loading embeddings and texts
    data = load_embeddings_and_texts(embeddings_filename)

    while True:
        question = input("> ")
        if question in ("quit", ""):
            break

        try:
            # Generate question embedding
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=[question]
            )
            question_embedding = response['data'][0]["embedding"]
        except Exception as e:
            print(f"Error: {e}")
            continue

        # Calculate similarity for each document
        similarities = [calculate_similarity(question_embedding, item['embedding']) for item in data]

        # Find the most similar document
        max_index = similarities.index(max(similarities))
        original_text = data[max_index]['text']

        # Generate response based on the most similar document
        answer = generate_response(original_text, question, company_name)
        print("\n\033[32mSupport:\033[0m")
        print(f"\033[32m{answer}\033[0m")

    print("Goodbye! Come back if you have any more questions. :)")


# Function to generate response based on the most similar article and given question
def generate_response(article_text, question, company_name):
    system_prompt = f"""You are an AI assistant. You work for #{company_name}. You will be asked questions from a
        customer and will answer in a helpful and friendly manner.
        
        You will be provided company information from #{company_name} under the
        [Article] section. The customer question will be provided under the
        [Question] section. You will answer the customers questions based on the
        article. Only provide the answer to the query don't respond with completed part of question.
        Answer in points and not in long paragraphs"""

    question_prompt = f"[Article]\n{article_text}\n\n[Question]\n{question}"

    try:

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        answer = response['choices'][0]['message']['content']
        return answer.lstrip()  # Remove leading whitespace
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    chat()
