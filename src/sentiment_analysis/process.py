import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    StringType,
    FloatType,
    StructType,
    StructField,
)
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from .load import load_model


# Define schema for sentiment analysis results
sentiment_schema = StructType(
    [
        StructField("review_text", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("score", FloatType(), True),
    ]
)
bc_tokenizer, bc_model = None, None


@pandas_udf(sentiment_schema)
def batch_sentiment_analysis(review_texts: pd.Series) -> pd.DataFrame:
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
    print(f"Processing {len(review_texts)} reviews...")
    tokenizer, model = bc_tokenizer.value, bc_model.value
    if tokenizer is None or model is None:
        print("Loading tokenizer and model since they are uninitialized...")
        # Load the tokenizer and model
        tokenizer, model = load_model()
    model.eval()  # Set model to evaluation mode

    results = []
    print("Processing reviews batch...")
    # Process each review
    for review_text in review_texts:

        try:
            # Tokenize and prepare model input
            inputs = tokenizer(
                review_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,  # Adjust based on average review length
            )

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
                    "review_text": review_text,
                    "sentiment": sentiment_label,
                    "score": float(confidence),
                }
            )

        except Exception as e:
            # Log error and continue with neutral sentiment
            print(f"Error processing review: {e}")
            results.append(
                {"review_text": review_text, "sentiment": "ERROR", "score": 0.0}
            )

    print(f"Done processing {len(results)} reviews..")
    # Create and return DataFrame with results
    return pd.DataFrame(results)
