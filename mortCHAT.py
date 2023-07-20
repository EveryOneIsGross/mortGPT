import datetime
import json
import math
import os
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import dotenv
import openai

# Initialize sentiment analyzer, keyword extractor, sentence transformer
sentiment_analyzer = SentimentIntensityAnalyzer()
keyword_extractor = Rake()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the existing memory if it exists
try:
    with open('memory.json', 'r') as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = {
        'conversations': [],
        'total_token_count': 0,
        'clock': 0,  # Represents seconds passed in 24-hour cycle (86400 seconds in a day)
        'last_interaction_time': datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),  # Added new field
    }

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

while True:
    # Get user input
    user_message = input("User: ")

    # Count the number of tokens in the user's message
    user_token_count = len(word_tokenize(user_message))

    # Update the total token count and clock
    memory['total_token_count'] += user_token_count
    memory['clock'] = (memory['clock'] + memory['total_token_count'] // 10) % 86400

    # Extract sentiment and keywords from the user's message
    user_sentiment = sentiment_analyzer.polarity_scores(user_message)
    keyword_extractor.extract_keywords_from_text(user_message)
    user_keywords = list(set(keyword_extractor.get_ranked_phrases()))  # Get keywords and convert to list

    # Check user input against stored previous topics
    previous_thought = None
    for conversation in memory['conversations']:
        if set(user_keywords).intersection(set(conversation['user_keywords'])):
            previous_thought = conversation['assistant_message']
            break

    # Generate sentence embedding for user's message
    user_embedding = model.encode([user_message])[0].tolist()  # Convert to list

    # Get the chatbot's current time in UTC format
    chatbot_time = datetime.datetime.utcnow()

    # Convert the user's last interaction time string to a datetime object
    user_time = datetime.datetime.strptime(memory['last_interaction_time'], "%Y-%m-%d %H:%M:%S")

    # Calculate the time difference in seconds
    time_difference_seconds = (chatbot_time - user_time).total_seconds()

    # Prepare conversation history
    conversation_history = []

    # Add conversation history from memory
    for conversation in memory['conversations']:
        conversation_history.append({"role": "user", "content": conversation['user_message']})
        conversation_history.append({"role": "assistant", "content": conversation['assistant_message']})

    # Construct system message
    system_message = f"You are a helpful assistant. The current time is {memory['clock']} token-seconds. Remember, time will move forward after this conversation. You have {86400 - memory['clock']} token-seconds left before the end of the day. The current time is {chatbot_time.strftime('%H:%M')}. The relative time difference between us is {time_difference_seconds} seconds."

    # Combine system message with user message
    user_message = system_message + " " + user_message
    if previous_thought:
        user_message += f" Previously, you mentioned something related: I am "

    # Add user's current message to conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Use the chat model to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=conversation_history
    )

    # Extract the assistant's message, sentiment, and keywords
    assistant_message = response['choices'][0]['message']['content']
    assistant_token_count = len(word_tokenize(assistant_message))
    assistant_sentiment = sentiment_analyzer.polarity_scores(assistant_message)
    keyword_extractor.extract_keywords_from_text(assistant_message)
    assistant_keywords = list(set(keyword_extractor.get_ranked_phrases())) if keyword_extractor.get_ranked_phrases() is not None else None  # Get keywords and convert to list



    # Update the total token count and clock again
    memory['total_token_count'] += assistant_token_count
    memory['clock'] = (memory['clock'] + memory['total_token_count'] // 10) % 86400

    # Generate sentence embedding for assistant's message
    assistant_embedding = model.encode([assistant_message])[0].tolist()  # Convert to list

    # Calculate cosine similarity between user's time and chatbot's time
    user_angle = (user_time.hour * 15) + (user_time.minute * 0.25) + (user_time.second * 0.00416667)
    chatbot_angle = (chatbot_time.hour * 15) + (chatbot_time.minute * 0.25) + (chatbot_time.second * 0.00416667)
    user_angle = np.array(user_angle).reshape(1, -1)
    chatbot_angle = np.array(chatbot_angle).reshape(1, -1)
    similarity = cosine_similarity(user_angle, chatbot_angle)[0][0]

    # Store the conversation in the memory, including the current "time"
    memory['conversations'].append({
        'user_message': user_message,
        'assistant_message': assistant_message,
        'user_sentiment': user_sentiment,
        'assistant_sentiment': assistant_sentiment,
        'user_keywords': user_keywords,
        'assistant_keywords': assistant_keywords,
        'user_embedding': user_embedding,
        'assistant_embedding': assistant_embedding,
        'user_token_count': user_token_count,
        'assistant_token_count': assistant_token_count,
        'similarity': similarity,
        'time': time_difference_seconds,  # Relative time difference in seconds
    })

    # Print assistant's response
    print("Assistant:", assistant_message)

    # After each interaction, update 'last_interaction_time' with the current time
    memory['last_interaction_time'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Save the memory
    try:
        with open('memory.json', 'w') as f:
            json.dump(memory, f)
    except TypeError as e:
        print(f"Error while saving memory: {e}")
