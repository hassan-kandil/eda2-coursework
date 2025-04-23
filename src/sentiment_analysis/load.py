from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType,
    FloatType,
    StructType,
    StructField,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import logger


def load_amazon_reviews(spark, file_path, sample_ratio=None):
    """
    Load Amazon reviews from a JSON Lines file.
    Optionally sample a fraction of the data for testing.
    """

    # Define the input schema for the JSON file
    input_schema = StructType(
        [
            StructField("text", StringType(), True),
            StructField("asin", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("rating", FloatType(), True),
            StructField("title", StringType(), True),
        ]
    )

    # Load the dataset with the predefined schema
    reviews_df = spark.read.schema(input_schema).json(file_path)

    # Sample if ratio provided
    if sample_ratio and 0 < sample_ratio < 1:
        reviews_df = reviews_df.sample(withReplacement=False, fraction=sample_ratio, seed=42)

    # Cache the dataframe to improve performance
    reviews_df.cache()

    return reviews_df


def load_model():
    """Load the pre-trained BERTweet model for sentiment analysis."""
    logger.info("Loading BERTweet model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis", cache_dir="/tmp/hf_cache"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis", cache_dir="/tmp/hf_cache"
    )
    # Move model to CPU as default (will be used across workers)
    model.cpu()
    return tokenizer, model
