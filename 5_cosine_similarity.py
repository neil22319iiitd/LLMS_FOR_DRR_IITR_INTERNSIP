import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# File paths
tweets_file = 'week6_manual_tweet.txt'  # File with all 3800 tweets, one per line
output_dir = 'WEEK6_CIF'  # Directory to store the output file
output_file = os.path.join(output_dir, 'WEEK6_CIF4.txt')  # Output file path

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the embeddings of the single tweet from the first file
single_tweet_embedding_file = 'embedding1_week6.csv'
single_tweet_embedding = pd.read_csv(single_tweet_embedding_file, header=None).values

# Load the embeddings of the 3800 tweets
embeddings_file = 'week6_embeddings_llama2-13b_batched3.csv'  # Update with your actual file name
tweet_embeddings = pd.read_csv(embeddings_file, header=None).values

# Compute cosine similarity
cos_similarities = cosine_similarity(single_tweet_embedding, tweet_embeddings)[0]

# Get the indices of the top 150 most similar tweets
top_150_indices = np.argsort(cos_similarities)[-250:][::-1]

# Extract the embeddings of these 150 tweets
top_150_embeddings = tweet_embeddings[top_150_indices]

# Load the embeddings of the single tweet from the second file
single_tweet_embedding_file1 = 'embedding2_week6.csv'
single_tweet_embedding1 = pd.read_csv(single_tweet_embedding_file1, header=None).values

# Compute cosine similarity between the second single tweet and these 150 tweet embeddings
cos_similarities1 = cosine_similarity(single_tweet_embedding1, top_150_embeddings)[0]

# Get the indices of the top 60 most similar tweets from the first segment
top_60_indices = np.argsort(cos_similarities1)[-60:][::-1]

# Extract the embeddings of these 60 tweets
top_60_embeddings = top_150_embeddings[top_60_indices]

# Load the embeddings of the single tweet from the third file
single_tweet_embedding_file2 = 'embedding3_week6.csv'
single_tweet_embedding2 = pd.read_csv(single_tweet_embedding_file2, header=None).values

# Compute cosine similarity between the third single tweet and these 60 tweet embeddings
cos_similarities2 = cosine_similarity(single_tweet_embedding2, top_60_embeddings)[0]

# Get the indices of the top 10 most similar tweets from the second segment
top_10_indices = np.argsort(cos_similarities2)[-10:][::-1]

# Convert to list and map to original indices
top_10_line_numbers = [top_60_indices[i] for i in top_10_indices]

# Read the tweets from the file
with open(tweets_file, 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Extract the top 10 tweets
top_10_tweets = [tweets[i].strip() for i in top_10_line_numbers]

# Print the top 10 tweets to the terminal
print("Top 10 Tweets:")
for tweet in top_10_tweets:
    print(tweet)

# Store the top 10 tweets in a new file
with open(output_file, 'w', encoding='utf-8') as file:
    for tweet in top_10_tweets:
        file.write(tweet + '\n')

print(f"\nTop 10 tweets have been saved to {output_file}")
