import time

from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
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
    input_path, output_path = "/Subscription_Boxes.jsonl", "/analysis_outputs"
    # Initialize Spark session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    # Load the dataset
    reviews_df = load_amazon_reviews(spark, input_path)
    # Count total reviews
    total_reviews = reviews_df.count()
    logger.info(f"Processing {total_reviews:,} reviews")
    # Load model and tokenizer
    tokenizer, model = load_model()
    # Broadcast model and tokenizer to all workers
    process.bc_tokenizer = spark.sparkContext.broadcast(tokenizer)
    process.bc_model = spark.sparkContext.broadcast(model)
    sentiment_results_df = reviews_df.withColumn(
        "result",
        process.batch_sentiment_analysis(reviews_df["text"]),
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
    sentiment_results_df.write.option("header", "true").mode("overwrite").csv(
        output_path
    )
    end_time = time.time()
    logger.info(f"Done processing all reviews in {end_time - start_time:.2f} seconds")
    # Combine results into a single csv file
    merge_results_csv_in_hdfs(
        output_path, "/summary_outputs", "sentiment_analysis_results"
    )
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
