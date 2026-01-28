"""Clinical indication dataset generation."""

from pathlib import Path

from loguru import logger


def clinical_indication(source: Path, destination: Path) -> None:
    """Generate clinical indication dataset from clinical report data.

    Args:
        source: Path to clinical report data (output/clinical_report)
        destination: Path to write clinical indication data (output/clinical_indication)

    Raises:
        NotImplementedError: This step is not yet implemented.
    """
    logger.info(f'Source path: {source}')
    logger.info(f'Destination path: {destination}')
    raise NotImplementedError('clinical_indication step is not yet implemented')
