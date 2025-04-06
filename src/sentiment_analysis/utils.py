# Generate random UUID function
from subprocess import PIPE, Popen
import uuid

from config import logger


def run_command(cmd: str):
    command_string = " ".join(cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    # Decode the byte output to string
    logger.info(f"Command {command_string} Output: {out.decode('utf-8')}")

    if err:
        logger.error(f"Command {command_string} Error: {err.decode('utf-8')}")


def generate_uuid():
    return str(uuid.uuid4())
