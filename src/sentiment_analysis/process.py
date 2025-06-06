import pandas as pd
import torch

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    StringType,
    FloatType,
    IntegerType,
    StructType,
    StructField,
)
from pyspark.sql.functions import col
from .load import load_model
from .config import logger


# Define schema for sentiment analysis results
sentiment_schema = StructType(
    [
        StructField("token_count", IntegerType(), True),
        StructField("sentiment", StringType(), True),
        StructField("score", FloatType(), True),
    ]
)
bc_tokenizer, bc_model = None, None


def _set_torch_threading(num_threads: int = 1):
    """
    Set the number of threads for PyTorch to use.
    Args:
        num_threads: Number of threads to use (default is 1)
    """
    current_num_threads, current_num_interop_threads = (
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )
    logger.info(
        f"Current PyTorch threading settings: {current_num_threads} threads, {current_num_interop_threads} interop threads"
    )
    if current_num_threads != num_threads:
        torch.set_num_threads(num_threads)
        logger.info(f"Set PyTorch to use {num_threads} threads.")
    if current_num_interop_threads != num_threads:
        torch.set_num_interop_threads(num_threads)
        logger.info(f"Set PyTorch to use {num_threads} interop threads.")

    logger.info(
        f"Updated PyTorch threading settings: {torch.get_num_threads()} threads, {torch.get_num_interop_threads()} interop threads"
    )


@pandas_udf(sentiment_schema)
def _batch_sentiment_analysis(review_texts: pd.Series) -> pd.DataFrame:
    """
    Process batches of reviews for sentiment analysis using pandas UDF.
    This function will be executed on each worker node.

    Args:
        tokenizer: The HuggingFace tokenizer for preparing inputs
        model: The pre-trained sentiment analysis model
        review_texts: Pandas Series containing review texts to analyze

    Returns:
        DataFrame with sentiment analysis results
    """
    global bc_tokenizer, bc_model
    logger.info(f"Processing {len(review_texts)} reviews...")
    if bc_tokenizer is None or bc_model is None:
        logger.info("Loading tokenizer and model since they are uninitialized...")
        # Load the tokenizer and model
        tokenizer, model = load_model()
    else:
        tokenizer, model = bc_tokenizer.value, bc_model.value
    model.eval()  # Set model to evaluation mode

    # Set the number of threads for PyTorch
    _set_torch_threading(num_threads=1)

    results = []
    total_tokens = 0
    # Process each review
    for review_text in review_texts:

        try:
            # Tokenize and prepare model input
            inputs = tokenizer(
                review_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,  # Bertweet max token length is 128
            )

            # Count tokens excluding padding using attention mask
            token_count = inputs["attention_mask"].sum().item()
            total_tokens += token_count

            # Run model inference
            with torch.no_grad():
                outputs = model(**inputs)
                sentiment_logits = outputs.logits

            # Get prediction
            predicted_class_id = sentiment_logits.argmax().item()
            sentiment_label = model.config.id2label[predicted_class_id]

            # Calculate confidence score (probability of predicted class)
            probs = torch.nn.functional.softmax(sentiment_logits, dim=1)
            confidence = probs[0][predicted_class_id].item()

            results.append(
                {
                    "token_count": int(token_count),
                    "sentiment": sentiment_label,
                    "score": float(confidence),
                }
            )

        except Exception as e:
            # Log error and continue with neutral sentiment
            logger.exception(f"Error processing review: {e}")
            results.append({"token_count": 0, "sentiment": "ERROR", "score": 0.0})
            pass

    # Log token statistics
    avg_tokens = total_tokens / len(review_texts) if review_texts.size > 0 else 0
    logger.info(f"Processed batch: {len(review_texts)} reviews with {total_tokens} tokens (avg: {avg_tokens:.1f})")

    # Create and return DataFrame with results
    return pd.DataFrame(results)


def process_reviews(
    reviews_df: pd.DataFrame,
    output_path: str,
    review_text_column: str = "preprocessed_text",
) -> pd.DataFrame:
    """
    Process reviews DataFrame for sentiment analysis and save results to output path.

    Args:
        reviews_df: DataFrame containing reviews
        output_path: Path to save the processed results
        review_text_column: Column name containing the reviews text
    """
    logger.info("Processing reviews for sentiment analysis...")
    # Apply batch sentiment analysis
    sentiment_results_df = reviews_df.withColumn(
        "result",
        _batch_sentiment_analysis(reviews_df[review_text_column]),
    )
    # Flatten the result column
    sentiment_results_df = sentiment_results_df.select(
        "*",  # Include all existing columns
        col("result.token_count").alias("token_count"),
        col("result.sentiment").alias("sentiment"),
        col("result.score").alias("score"),
    ).drop("result")
    # Cache dataframe to prevent recomputations
    logger.info("Caching results dataframe...")
    sentiment_results_df = sentiment_results_df.cache()
    # Save results to output path in CSV format (quote and escape options)
    # fmt: off
    sentiment_results_df.write.format("csv")\
        .option("header", "true")\
        .option("quote", '"')\
        .option("escape", '"')\
        .option("quoteMode", "MINIMAL")\
        .option("emptyValue", "")\
        .mode("overwrite")\
        .save(output_path)
    # fmt: on
    logger.info(f"Sentiment analysis results saved to {output_path}")
    return sentiment_results_df
