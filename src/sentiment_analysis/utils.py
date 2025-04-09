# Generate random UUID function
import os
from subprocess import PIPE, Popen
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
