import os
import matplotlib.pyplot as plt
import pandas as pd

from .utils import (
    compress_directory,
    delete_local_directory,
    delete_local_file,
    logger,
    run_command,
    upload_file_to_hdfs,
    write_df_to_hdfs_csv,
)
from pyspark.sql.functions import col, round as spark_round, mean, stddev, min, max, lit, sum


def merge_results_csv_in_hdfs(read_hdfs_path: str, write_hdfs_path: str, csv_file_name: str) -> None:
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
    upload_file_to_hdfs(temp_csv_path, final_path)
    # Remove the local merged file
    delete_local_file(temp_csv_path)
    logger.info(f"Successfully wrote {csv_file_name} to HDFS at {final_path}")


def _plot_sentiment_distribution_bar(
    sentiments: list[str], counts: list[int], colors: list[str], local_path: str
) -> None:
    """
    Helper function to plot sentiment distribution as a bar chart.

    Args:
        sentiments: List of sentiment labels
        counts: List of counts for each sentiment
        colors: List of colors for each sentiment
    """
    # Sentiment distribution plot
    plt.figure(figsize=(10, 6))
    plt.bar(sentiments, counts, color=colors)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(local_path, dpi=300)
    logger.info(f"Saved sentiment distribution visualization to {local_path}")


def _plot_sentiment_distribution_pie(
    sentiments: list[str],
    counts: list[int],
    percentages: list[float],
    colors: list[str],
    local_path: str,
) -> None:
    """
    Helper function to plot sentiment distribution as a pie chart.

    Args:
        sentiments: List of sentiment labels
        counts: List of counts for each sentiment
        percentages: List of percentages for each sentiment
        colors: List of colors for each sentiment
    """
    plt.figure(figsize=(8, 8))
    # Create cleaner labels with percentages
    pie_labels = [f"{sentiment} ({pct:.1f}%)" for sentiment, pct in zip(sentiments, percentages)]

    # Draw the pie chart
    plt.pie(
        counts,
        labels=pie_labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=False,  # Remove shadow for cleaner look
        startangle=90,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1,
        },  # Add white edges between slices
    )
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis("equal")
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(local_path, dpi=300)  # Higher DPI for better quality
    logger.info(f"Saved sentiment pie chart to {local_path}")


def plot_sentiment_results(sentiment_results: list[dict], output_path: str) -> None:
    # Extract data for visualization
    sentiments = [row["sentiment"] for row in sentiment_results]
    counts = [row["count"] for row in sentiment_results]
    percentages = [row["percentage"] for row in sentiment_results]
    colors = []

    # Define color mapping
    color_map = {"NEG": "red", "NEU": "gray", "POS": "green"}

    # Assign colors based on sentiment
    colors.extend([color_map.get(sentiment, "blue") for sentiment in sentiments])

    # Plot sentiment distribution bar chart
    local_sentiment_distr_chart_path = "/tmp/sentiment_distribution.png"
    hdfs_sentiment_distr_chart_path = f"{output_path}/sentiment_distribution.png"
    _plot_sentiment_distribution_bar(sentiments, counts, colors, local_sentiment_distr_chart_path)

    # Upload the sentiment distribution plot to HDFS
    upload_file_to_hdfs(local_sentiment_distr_chart_path, hdfs_sentiment_distr_chart_path)
    logger.info("Uploaded sentiment distribution bar char to HDFS")

    # Plot sentiment distribution pie chart
    local_sentiment_pie_chart_path = "/tmp/sentiment_pie_chart.png"
    hdfs_sentiment_pie_chart_path = f"{output_path}/sentiment_pie_chart.png"
    _plot_sentiment_distribution_pie(sentiments, counts, percentages, colors, local_sentiment_pie_chart_path)

    # Upload the sentiment pie chart to HDFS
    upload_file_to_hdfs(local_sentiment_pie_chart_path, hdfs_sentiment_pie_chart_path)
    logger.info("Uploaded sentiment distribution pie chart to HDFS")


def generate_sentiment_statistics(sentiment_analysis_results_df: pd.DataFrame, summary_output_path: str) -> None:
    """
    Generate statistics on sentiment analysis results and plot visualizations.

    Args:
        sentiment_df: Spark DataFrame containing sentiment analysis results
        summary_output_path: HDFS path to save the summary output
    """
    logger.info("Generating sentiment analysis statistics...")

    # Count total reviews
    total_count = sentiment_analysis_results_df.count()
    logger.info(f"Total processed reviews: {total_count:,}")

    # Score statistics dataframe
    # Calculate mean, stddev, min, and max confidence scores
    score_stats_df = sentiment_analysis_results_df.select(
        lit(total_count).alias("total_reviews"),
        mean("score").alias("mean_confidence"),
        stddev("score").alias("stddev_confidence"),
        min("score").alias("min_confidence"),
        max("score").alias("max_confidence"),
    )
    score_stats = score_stats_df.collect()[0]

    # Log the statistics
    logger.info("Confidence Score Statistics:")
    logger.info(f"  Mean: {score_stats['mean_confidence']:.4f}")
    logger.info(f"  StdDev: {score_stats['stddev_confidence']:.4f}")
    logger.info(f"  Min: {score_stats['min_confidence']:.4f}")
    logger.info(f"  Max: {score_stats['max_confidence']:.4f}")

    # Write the score statistics DataFrame to HDFS in CSV format
    write_df_to_hdfs_csv(score_stats_df, summary_output_path, "sentiment_overall_stats")

    # Get sentiment distribution using Spark
    sentiment_counts_df = sentiment_analysis_results_df.groupBy("sentiment").count().orderBy("sentiment")

    # Calculate percentages using Spark
    sentiment_with_pct_df = sentiment_counts_df.withColumn(
        "percentage", spark_round((col("count") / total_count) * 100, 2)
    )

    # Collect only the aggregated results
    sentiment_results = sentiment_with_pct_df.collect()

    # Log the distribution
    logger.info("Sentiment Distribution:")
    for row in sentiment_results:
        logger.info(f"  {row['sentiment']}: {row['count']} reviews ({row['percentage']}%)")

    # Write the summary DataFrames to HDFS in CSV format
    write_df_to_hdfs_csv(sentiment_with_pct_df, summary_output_path, "sentiment_distribution")

    # Plot sentiment distribution
    logger.info("Generating sentiment distribution visualizations...")
    plot_sentiment_results(sentiment_results, summary_output_path)
    logger.info("Sentiment analysis statistics generation complete.")


def generate_token_statistics(sentiment_analysis_results_df: pd.DataFrame, summary_output_path: str) -> dict:
    """
    Generate token statistics from sentiment analysis results.

    Args:
        sentiment_analysis_results_df: DataFrame containing sentiment analysis results
        summary_output_path: HDFS path to save the summary output
    """
    logger.info("Generating token statistics...")

    # Token statistics dataframe
    # Calculate mean, stddev, min, and max token counts
    token_stats_df = sentiment_analysis_results_df.agg(
        sum("token_count").alias("total_tokens"),
        mean("token_count").alias("mean_token_count"),
        stddev("token_count").alias("stddev_token_count"),
        max("token_count").alias("max_token_count"),
        min("token_count").alias("min_token_count"),
    )
    token_stats = token_stats_df.collect()[0]

    # Log the statistics
    logger.info("Token Count Statistics:")
    logger.info(f"  Total Tokens: {token_stats['total_tokens']}")
    logger.info(f"  Mean: {token_stats['mean_token_count']:.4f}")
    logger.info(f"  StdDev: {token_stats['stddev_token_count']:.4f}")
    logger.info(f"  Min: {token_stats['min_token_count']:.4f}")
    logger.info(f"  Max: {token_stats['max_token_count']:.4f}")

    # Write the token statistics DataFrame to HDFS in CSV format
    write_df_to_hdfs_csv(token_stats_df, summary_output_path, "sentiment_token_stats")

    return token_stats


def compress_hdfs_output_dir(hdfs_path):
    logger.info(f"Compressing HDFS output directory: {hdfs_path}")
    dir_name = os.path.basename(os.path.normpath(hdfs_path))
    tar_file_name = "_".join(hdfs_path.strip("/").split("/")) + ".tar.gz"
    hdfs_get_cmd = ["/home/almalinux/hadoop-3.4.0/bin/hdfs", "dfs", "-get", hdfs_path]
    run_command(hdfs_get_cmd)
    compress_directory(dir_name, tar_file_name)
    upload_file_to_hdfs(tar_file_name, "/")
    delete_local_directory(dir_name)
    delete_local_file(tar_file_name)
