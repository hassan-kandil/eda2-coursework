import pandas as pd
import torch

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    StringType,
    FloatType,
    StructType,
    StructField,
)
from pyspark.sql.functions import col
from .load import load_model
from .config import logger

from .utils import round_up_to_multiple


# Define schema for sentiment analysis results
sentiment_schema = StructType(
    [
        StructField("review_text", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("score", FloatType(), True),
    ]
)
bc_tokenizer, bc_model = None, None


def _set_torch_threading(num_threads=1):
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
    tokenizer, model = bc_tokenizer.value, bc_model.value
    if tokenizer is None or model is None:
        logger.info("Loading tokenizer and model since they are uninitialized...")
        # Load the tokenizer and model
        tokenizer, model = load_model()
    model.eval()  # Set model to evaluation mode

    # Set the number of threads for PyTorch
    _set_torch_threading(num_threads=1)

    results = []
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
            logger.error(f"Error processing review: {e}")
            results.append(
                {"review_text": review_text, "sentiment": "ERROR", "score": 0.0}
            )

    logger.info(f"Done processing {len(results)} reviews..")
    # Create and return DataFrame with results
    return pd.DataFrame(results)


def _get_hdfs_file_size_in_mb(sc, file_path):
    """
    Get the size of a file in MB.
    Args:
        file_path: Path to the file
    Returns:
        Size of the file in MB
    """
    # Get Hadoop configuration and file system
    hadoop_conf = sc._jsc.hadoopConfiguration()
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path = sc._jvm.org.apache.hadoop.fs.Path(file_path)

    # Get file status and size
    file_status = fs.getFileStatus(path)
    file_size_bytes = file_status.getLen()  # size in bytes
    file_size_mb = file_size_bytes / (1024 * 1024)  # convert to MB

    return file_size_mb


def repartition_dataset(
    spark, input_file_path, reviews_df, target_partition_size_mb=128
):
    """
    Repartition the DataFrame to optimize for processing.
    This function repartitions the DataFrame based on the target partition size.

    Args:
        saprk: Spark Session
        input_file: Path to the input file
        reviews_df: DataFrame to be repartitioned
        target_partition_size_mb: Target size of each partition in MB

    Returns:
        Repartitioned DataFrame
    """
    logger.info("Repartitioning DataFrame for optimal processing...")
    # Get the size of the input file in MB
    file_size_mb = _get_hdfs_file_size_in_mb(spark.sparkContext, input_file_path)
    logger.info(f"Input file size: {file_size_mb:.2f} MB")
    # Calculate the number of partitions based on the target size
    num_partitions = int(file_size_mb / (target_partition_size_mb))

    # Round up to the nearest multiple of the minimum number of partitions
    total_cores = spark.sparkContext.defaultParallelism
    task_cpus = int(spark.conf.get("spark.task.cpus", "1"))
    min_partitions = total_cores // task_cpus
    num_partitions = round_up_to_multiple(num_partitions, min_partitions)

    # Repartition the DataFrame
    logger.info(
        f"Repartitioning DataFrame into {num_partitions} partitions based on {file_size_mb}MB file size and {target_partition_size_mb}MB target partition size..."
    )
    repartitioned_df = reviews_df.repartition(num_partitions)
    logger.info(
        f"Repartitioned DataFrame into {repartitioned_df.rdd.getNumPartitions()} partitions."
    )
    return repartitioned_df


def process_reviews(reviews_df, output_path):
    """
    Process reviews DataFrame for sentiment analysis and save results to output path.

    Args:
        reviews_df: DataFrame containing reviews
        output_path: Path to save the processed results
    """
    logger.info("Processing reviews for sentiment analysis...")
    # Apply batch sentiment analysis
    sentiment_results_df = reviews_df.withColumn(
        "result",
        _batch_sentiment_analysis(reviews_df["text"]),
    )
    # Flatten the result column
    sentiment_results_df = sentiment_results_df.select(
        col("asin"),
        col("user_id"),
        col("result.review_text"),
        col("result.sentiment"),
        col("result.score"),
    )
    # Save results to output path
    sentiment_results_df.write.option("header", "true").mode("overwrite").csv(
        output_path
    )
