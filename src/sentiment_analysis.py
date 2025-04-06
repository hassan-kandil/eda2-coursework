from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import json
import pandas as pd
import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

amazon_reviews = []
# Load the JSON data (assuming it's a list of review objects)
with open("Subscription_Boxes.jsonl", "r") as f:
    for line in f:
        amazon_reviews.append(json.loads(line))

amazon_reviews_text = [review["text"] for review in amazon_reviews]

# Initialize the DistilBERT tokenizer
bertweet_tokenizer = AutoTokenizer.from_pretrained(
    "finiteautomata/bertweet-base-sentiment-analysis"
)

# Compute token lengths for each review text
token_counts = [
    len(
        bertweet_tokenizer(
            review["text"], return_tensors="pt", padding="max_length", truncation=True
        )
    )
    for review in amazon_reviews
]
average_tokens = sum(token_counts) / len(token_counts)

print(f"Average token count: {average_tokens:.1f}")

bertweet_tokenizer = AutoTokenizer.from_pretrained(
    "finiteautomata/bertweet-base-sentiment-analysis"
)
bertweet_model = AutoModelForSequenceClassification.from_pretrained(
    "finiteautomata/bertweet-base-sentiment-analysis"
)

fine_tuned_sentiment_results = []
processing_times = []
token_counts = []

for review_text in tqdm(amazon_reviews_text, desc="Processing reviews"):
    start_time = time.time()

    # Tokenize the review text
    inputs = bertweet_tokenizer(
        review_text, return_tensors="pt", padding="max_length", truncation=True
    )

    # Calculate the actual number of tokens (using attention_mask to ignore padding)
    token_count = inputs["attention_mask"].sum().item()
    token_counts.append(token_count)

    # Perform sentiment analysis
    with torch.no_grad():
        sentiment_batch = bertweet_model(**inputs).logits

    predicted_class_id = sentiment_batch.argmax().item()
    sentiment_label = bertweet_model.config.id2label[predicted_class_id]

    fine_tuned_sentiment_results.append(
        {"review_text": review_text, "sentiment": sentiment_label}
    )

    batch_time = time.time() - start_time
    processing_times.append(batch_time)

# Compute average and maximum values
avg_time = sum(processing_times) / len(processing_times)
max_time = max(processing_times)
avg_tokens = sum(token_counts) / len(token_counts)
max_tokens = max(token_counts)
total_batches = len(processing_times)

print("Total number of batches:", total_batches)
print("Total time taken: {:.4f} seconds".format(sum(processing_times)))
print("Average time per batch: {:.4f} seconds".format(avg_time))
print("Maximum time for a batch: {:.4f} seconds".format(max_time))
print("Average batch size (in tokens): {:.2f}".format(avg_tokens))
print("Longest batch (in tokens):", max_tokens)


bertweet_sentiment_amazon_df = pd.DataFrame(fine_tuned_sentiment_results)

# Save the results to a CSV file
bertweet_sentiment_amazon_df.to_csv(
    "bertweet_sentiment_analysis_amazon_results.csv", index=False
)

print("Result saved to 'bertweet_sentiment_analysis_amazon_results.csv'.")
