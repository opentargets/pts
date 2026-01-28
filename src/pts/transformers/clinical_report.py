"""Clinical report dataset generation."""

from pathlib import Path

from loguru import logger


def clinical_report(source: dict[str, Path], destination: dict[str, Path]) -> None:
    """Generate clinical report dataset from ChEMBL molecules and disease data.

    Args:
        source: Dictionary containing paths to input data:
            - chembl_molecule: Path to ChEMBL molecule data (intermediate/chembl_molecule)
            - disease: Path to disease data (output/disease)
        destination: Dictionary containing paths to output data:
            - output: Path to write valid clinical reports (output/clinical_report)
            - excluded: Path to write invalid clinical reports (excluded/clinical_report)

    Raises:
        NotImplementedError: This step is not yet implemented.
    """
    logger.info(f'Source paths: {source}')
    logger.info(f'Destination paths: {destination}')
    raise NotImplementedError('clinical_report step is not yet implemented')
