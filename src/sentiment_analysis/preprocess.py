"""
Text preprocessing module for sentiment analysis.
Provides functions to clean and normalize text before analysis.
"""

import re
from pyspark.sql.functions import col, udf, length, trim
from pyspark.sql.types import StringType

from .config import logger
from .utils import round_up_to_multiple


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


def repartition_dataset(spark, input_file_path, reviews_df, target_partition_size_mb=128):
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
    num_partitions = max(1, int(file_size_mb / (target_partition_size_mb)))

    # Round up to the nearest multiple of the minimum number of partitions
    total_cores = spark.sparkContext.defaultParallelism
    task_cpus = int(spark.conf.get("spark.task.cpus", "1"))
    min_partitions = total_cores // task_cpus
    num_partitions = round_up_to_multiple(num_partitions, min_partitions)
    num_partitions = 15

    # Repartition the DataFrame
    logger.info(
        f"Repartitioning DataFrame into {num_partitions} partitions based on {file_size_mb:.2f}MB file size and {target_partition_size_mb}MB target partition size..."
    )
    repartitioned_df = reviews_df.repartition(num_partitions)
    logger.info(f"Repartitioned DataFrame into {repartitioned_df.rdd.getNumPartitions()} partitions.")
    return repartitioned_df


def _clean_text(text):
    """
    Clean and normalize text by removing special characters
    and normalizing whitespace. Preserves emojis.

    Args:
        text (str): The input text to clean

    Returns:
        str: Cleaned and normalized text
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace URLs with token
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # Replace user mentions with token
    text = re.sub(r"@\w+", "[USER]", text)

    # Replace product codes/numbers with token
    text = re.sub(r"\b[A-Z0-9]{10,}\b", "[PRODUCT]", text)

    # Replace multiple newlines with a single one
    text = re.sub(r"\n+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


@udf(returnType=StringType())
def _preprocess_text_udf(text):
    """
    UDF wrapper for the clean_text function to use with Spark DataFrames.

    Args:
        text (str): The input text to clean

    Returns:
        str: Cleaned and normalized text
    """

    return _clean_text(text)


def preprocess_reviews(df, text_column="text", output_column="preprocessed_text"):
    """
    Apply preprocessing to a text column in a Spark DataFrame.

    Args:
        df: Spark DataFrame containing text to preprocess
        text_column (str): Name of the column containing text
        output_column (str): Name of the column to store preprocessed text

    Returns:
        DataFrame: DataFrame with added preprocessed text column
    """
    logger.info("Starting review text preprocessing...")

    # Count original records
    original_count = df.count()
    logger.info(f"Original review count: {original_count:,}")

    # 1. Check for null or empty texts
    df = df.filter(col(text_column).isNotNull() & (length(trim(col(text_column))) > 0))

    # 2. Add text length column
    df = df.withColumn("text_length", length(col(text_column)))

    # 3. Apply the preprocessing UDF
    preprocessed_df = df.withColumn(output_column, _preprocess_text_udf(df[text_column]))

    # Cache the preprocessed DataFrame to improve performance
    preprocessed_df.cache()

    # Log preprocessing results
    final_count = preprocessed_df.count()
    rejected_count = original_count - final_count
    logger.info(f"Preprocessing complete: {final_count:,} reviews retained, {rejected_count:,} filtered out")
    logger.info(f"Rejection rate: {100 * rejected_count / original_count:.2f}%")

    return preprocessed_df
