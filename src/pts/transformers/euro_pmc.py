"""Parser to extract publication date and publication identifier from EuropePMC data dump."""

import subprocess
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.epmc import epmc_schema


def get_json_files_in_folder(source: Path) -> list[Path]:
    """Get list of JSON files in the input folder.

    Args:
        source (Path): input path to read files from.

    Returns:
        list[Path]: list of JSON files.

    Raises:
        ValueError if source is not a directory.
    """
    # Raise error if source is not a folder:
    if not source.is_dir():
        raise ValueError(f'Source path {source} is not a directory')

    # Get only JSON files
    json_files = [item for item in source.iterdir() if item.is_file() and item.suffix.lower() == '.json']
    logger.info(f'There are {len(json_files)} JSON files to process.')

    return json_files


def publication_date_extractor(source: Path, destination: Path) -> None:
    """Extract publication date and identifiers from EuropePMC data dump.

    Args:
        source (Path): souce directory where the JSON files are located.
        desctination (Path): destination location of parquet files.

    Raises:
        ValueError raised if destination is not a directory.
    """
    jq_query = '{ pmid, firstPublicationDate, dateOfPublication: .journalInfo.dateOfPublication, id, source, pmcid }'

    # Destination should be a directory:
    if not destination.is_dir():
        raise ValueError(f'Source path {destination} is not a directory')

    # Looping through all json files found in the source directory:
    for json_file in get_json_files_in_folder(source):
        base_name = json_file.name
        output_file_name = base_name.replace('json', 'parquet')
        temp_file = Path(f'{tempfile.gettempdir()}/{base_name}')

        # we will have to do it like this for now, until polars fixes https://github.com/pola-rs/polars/issues/17677
        jq = subprocess.run(
            ['jq', '-c', jq_query, str(json_file)],
            capture_output=True,
            text=True,
        )
        if jq.returncode != 0:
            logger.error(f'jq error: {jq.stderr}')
            raise OSError(f'jq error: {jq.stderr}')

        temp_file.write_text(jq.stdout)
        logger.info('transforming ndjson into parquet')

        # write the result locally
        pl.read_ndjson(temp_file, schema=epmc_schema).write_parquet(destination.joinpath(output_file_name))
        logger.info('transformation complete')
