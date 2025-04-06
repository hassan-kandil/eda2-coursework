import time

from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sentiment_analysis.config import logger
from sentiment_analysis.utils import run_command
from sentiment_analysis.load import load_amazon_reviews, load_model
from sentiment_analysis.process import batch_sentiment_analysis


def write_df_to_hdfs_csv(df, hdfs_path, csv_file_name):
    logger.info(f"WRITING ANALYSIS SUMMARY OUTPUT {csv_file_name} TO HDFS...")
    write_path = hdfs_path + csv_file_name
    df.write.option("header", "true").mode("overwrite").csv(write_path)
    hdfs_mv_cmd = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-mv",
        write_path + "/part-00000-*.csv",
        write_path + ".csv",
    ]
    run_command(hdfs_mv_cmd)
    hdfs_rm_cmd = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-rm",
        "-r",
        write_path,
    ]
    run_command(hdfs_rm_cmd)


if __name__ == "__main__":
    input_path, output_path = "/Subscription_Boxes.jsonl", "/output"
    # Initialize Spark session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    # Load the dataset
    reviews_df = load_amazon_reviews(spark, input_path, sample_ratio=0.05)
    # Count total reviews
    total_reviews = reviews_df.count()
    print(f"Processing {total_reviews:,} reviews")
    # Load model and tokenizer
    tokenizer, model = load_model()
    # Broadcast model and tokenizer to all workers
    broadcast_tokenizer = spark.sparkContext.broadcast(tokenizer)
    broadcast_model = spark.sparkContext.broadcast(model)
    sentiment_results_df = reviews_df.withColumn(
        "result",
        batch_sentiment_analysis(
            broadcast_tokenizer.value, broadcast_model.value, reviews_df["text"]
        ),
    )
    # Flatten the result column
    sentiment_results_df = sentiment_results_df.select(
        col("asin"),
        col("user_id"),
        col("result.review_text"),
        col("result.sentiment"),
        col("result.score"),
    )
    start_time = time.time()
    sentiment_results_df.write.mode("overwrite").parquet(output_path)
    end_time = time.time()
    print(f"Done processing all reviews in {end_time - start_time:.2f} seconds")
    # Combine results into a single csv file
    # Read the Parquet files
    df = spark.read.parquet(output_path).coalesce(1)
    # Write to CSV
    write_df_to_hdfs_csv(
        df,
        "/summary_outputs/",
        "sentiment_analysis_results",
    )

    # Stop Spark session
    spark.stop()
