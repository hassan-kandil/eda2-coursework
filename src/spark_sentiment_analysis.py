import time

from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from sentiment_analysis.config import logger
from sentiment_analysis.utils import delete_local_file, run_command
from sentiment_analysis.load import load_amazon_reviews, load_model
import sentiment_analysis.process as process


def merge_results_csv_in_hdfs(read_hdfs_path, write_hdfs_path, csv_file_name):
    logger.info(f"WRITING ANALYSIS SUMMARY OUTPUT {csv_file_name} TO HDFS...")
    final_path = f"{write_hdfs_path}/{csv_file_name}.csv"
    temp_csv_path = "/tmp/merged_results.csv"
    # Merge csv files from hdfs and save them locally
    merge_command = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-getmerge",
        f"{read_hdfs_path}/part-*.csv",
        temp_csv_path,
    ]
    run_command(merge_command)

    # Upload the merged csv to hdfs
    upload_command = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-put",
        "-f",
        temp_csv_path,
        final_path,
    ]
    run_command(upload_command)
    # Remove the local merged file
    delete_local_file(temp_csv_path)
    logger.info(f"Successfully wrote {csv_file_name} to HDFS at {final_path}")


if __name__ == "__main__":
    input_path, output_path = "/Subscription_Boxes.jsonl", "/analysis_outputs"
    # Initialize Spark session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    # Load the dataset
    reviews_df = load_amazon_reviews(spark, input_path)
    # Count total reviews
    total_reviews = reviews_df.count()
    logger.info(f"Starting to process all {total_reviews:,} reviews")
    # Load model and tokenizer
    tokenizer, model = load_model()
    # Broadcast model and tokenizer to all workers
    if process.bc_model is None:
        logger.info("Broadcasting model and tokenizer to all workers")
        process.bc_tokenizer = spark.sparkContext.broadcast(tokenizer)
        process.bc_model = spark.sparkContext.broadcast(model)

    start_time = time.time()
    process.process_reviews(reviews_df=reviews_df, output_path=output_path)
    end_time = time.time()
    logger.info(f"Done processing all reviews in {end_time - start_time:.2f} seconds")
    # Combine results into a single csv file
    merge_results_csv_in_hdfs(
        output_path, "/summary_outputs", "sentiment_analysis_results"
    )
    # Stop Spark session
    spark.stop()
