# Generate random UUID function
import os
import math

import shutil
from subprocess import PIPE, Popen
import tarfile
import uuid

from .config import logger


def run_command(cmd: str):
    command_string = " ".join(cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    # Decode the byte output to string
    logger.info(f"Command {command_string} Output: {out.decode('utf-8')}")

    if err:
        logger.error(f"Command {command_string} Error: {err.decode('utf-8')}")


def delete_local_file(file_path):
    try:
        os.remove(file_path)
        logger.info(f"{file_path} local file has been deleted.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
        raise
    except PermissionError:
        logger.error(f"Permission denied to delete {file_path}.")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def generate_uuid():
    return str(uuid.uuid4())


def round_up_to_multiple(n, multiple):
    return int(math.ceil(n / multiple) * multiple)


def write_df_to_hdfs_csv(df, hdfs_path, csv_file_name):
    logger.info(f"WRITING ANALYSIS SUMMARY OUTPUT {csv_file_name} TO HDFS...")
    write_path = f"{hdfs_path}/{csv_file_name}"
    df.coalesce(1).write.option("header", "true").mode("overwrite").csv(write_path)
    delete_from_hdfs(write_path + ".csv")
    hdfs_mv_cmd = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-mv",
        write_path + "/part-00000-*.csv",
        write_path + ".csv",
    ]
    run_command(hdfs_mv_cmd)
    delete_from_hdfs(write_path)
    logger.info(f"Successfully wrote {csv_file_name} to HDFS at {hdfs_path}")


def delete_from_hdfs(hdfs_path):
    logger.info(f"DELETING FILE FROM HDFS {hdfs_path}...")
    delete_command = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-rm",
        "-r",
        hdfs_path,
    ]
    run_command(delete_command)


def upload_file_to_hdfs(local_file_path, hdfs_path):
    logger.info(f"UPLOADING {local_file_path} TO HDFS...")
    upload_command = [
        "/home/almalinux/hadoop-3.4.0/bin/hdfs",
        "dfs",
        "-put",
        "-f",
        local_file_path,
        hdfs_path,
    ]
    run_command(upload_command)
    logger.info(f"Successfully uploaded {local_file_path} to HDFS at {hdfs_path}")


def compress_directory(directory_path, tar_gz_path):
    # Compress the directory into a tar.gz file
    logger.info(f"Compressing directory {directory_path} into {tar_gz_path}")
    with tarfile.open(tar_gz_path, "w:gz") as tar:
        tar.add(directory_path, arcname=".")
    logger.info(f"Directory {directory_path} compressed into {tar_gz_path}")


def delete_local_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        logger.info(f"{directory_path} local directory has been deleted.")
    except FileNotFoundError:
        logger.error(f"{directory_path} does not exist.")
        raise
    except PermissionError:
        logger.error(f"Permission denied to delete {directory_path}.")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
