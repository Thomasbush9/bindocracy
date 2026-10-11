"""Non-destructive directory publication on POSIX filesystems, including Lustre."""
from contextlib import suppress
from pathlib import Path


def publish_directory(stage: Path, destination: Path) -> None:
    """Reserve exclusively, then atomically expose a completed sibling tree.

    Readers may see an empty reservation, never a partially populated tree.
    A crash can leave that reservation; retries refuse it rather than adopting it.
    The parent directory must be trusted against hostile replacement races.
    """
    destination.mkdir(mode=0o700)
    reserved = destination.stat()
    try:
        stage.rename(destination)
    except BaseException:
        # Preserve both the original failure and any intervening files/replacement.
        with suppress(OSError):
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (reserved.st_dev, reserved.st_ino):
                destination.rmdir()
        raise
