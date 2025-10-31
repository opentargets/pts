from enum import Enum
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql import Column, DataFrame

if TYPE_CHECKING:
    from enum import Enum


class GenerateDiseaseCellLines:
    """Generate "diseaseCellLines" object from a cell passport file.

    !!!
    There's one important bit here: I have noticed that we frequenty get cell line names
    with missing dashes. Therefore the cell line names are cleaned up by removing dashes.
    It has to be done when joining with other datasets.
    !!!

    Args:
        cell_passport_file: Path to the cell passport file.
    """

    def __init__(
        self,
        cell_passport_data: DataFrame,
        cell_line_to_uberon_mapping: DataFrame,
    ) -> None:
        self.cell_passport_data = cell_passport_data
        self.tissue_to_uberon_map = cell_line_to_uberon_mapping

    def generate_disease_cell_lines(self) -> DataFrame:
        """Reading and procesing cell line data from the cell passport file.

        The schema of the returned dataframe is:

        root
        |-- name: string (nullable = true)
        |-- id: string (nullable = true)
        |-- biomarkerList: array (nullable = true)
        |    |-- element: struct (containsNull = true)
        |    |    |-- name: string (nullable = true)
        |    |    |-- description: string (nullable = true)
        |-- diseaseCellLine: struct (nullable = false)
        |    |-- tissue: string (nullable = true)
        |    |-- name: string (nullable = true)
        |    |-- id: string (nullable = true)
        |    |-- tissueId: string (nullable = true)

        Note:
            * Microsatellite stability is the only inferred biomarker.
            * The cell line name has the dashes removed.
            * Id is the cell line identifier from Sanger
            * Tissue id is the UBERON identifier for the tissue, based on manual curation.
        """
        cell_df = (
            self.cell_passport_data.select(
                f.col('model_name').alias('name'),
                f.col('model_id').alias('id'),
                f.lower(f.col('tissue')).alias('tissueFromSource'),
                f.array(self.parse_msi_status(f.col('msi_status'))).alias('biomarkerList'),
            )
            .join(self.tissue_to_uberon_map, on='tissueFromSource', how='left')
            .persist()
        )
        return cell_df.select(
            f.regexp_replace(f.col('name'), '-', '').alias('name'),
            'id',
            'biomarkerList',
            f.struct(f.col('tissueName').alias('tissue'), 'name', 'id', 'tissueId').alias('diseaseCellLine'),
        )

    @staticmethod
    def parse_msi_status(status: Column) -> Column:
        """Based on the content of the MSI status, we generate the corresponding biomarker object."""
        return f.when(
            status == 'MSI',
            f.struct(
                f.lit('MSI').alias('name'),
                f.lit('Microsatellite instable').alias('description'),
            ),
        ).when(
            status == 'MSS',
            f.struct(
                f.lit('MSS').alias('name'),
                f.lit('Microsatellite stable').alias('description'),
            ),
        )


def update_quality_flag(qc: Column, flag_condition: Column, flag_text: Enum) -> Column:
    """Update the provided quality control list with a new flag if condition is met.

    Args:
        qc (Column): Array column with the current list of qc flags.
        flag_condition (Column): This is a column of booleans, signing which row should be flagged
        flag_text (Enum): Text for the new quality control flag

    Returns:
        Column: Array column with the updated list of qc flags.


    Examples:
        >>> s = "study STRING, qualityControls ARRAY<STRING>, condition BOOLEAN"
        >>> d =  [("S1", ["qc1"], True), ("S2", ["qc3"], False)]
        >>> df = spark.createDataFrame(d, s)
        >>> df.show()
        +-----+---------------+---------+
        |study|qualityControls|condition|
        +-----+---------------+---------+
        |   S1|          [qc1]|     true|
        |   S2|          [qc3]|    false|
        +-----+---------------+---------+
        <BLANKLINE>

        >>> class QC(Enum):
        ...     QC1 = "qc1"
        ...     QC2 = "qc2"
        ...     QC3 = "qc3"

        >>> condition = f.col("condition")
        >>> new_qc = update_quality_flag(f.col("qualityControls"), condition, QC.QC2)
        >>> df.withColumn("qualityControls", new_qc).show()
        +-----+---------------+---------+
        |study|qualityControls|condition|
        +-----+---------------+---------+
        |   S1|     [qc1, qc2]|     true|
        |   S2|          [qc3]|    false|
        +-----+---------------+---------+
        <BLANKLINE>
    """
    qc = f.when(qc.isNull(), f.array()).otherwise(qc)
    return f.when(
        flag_condition,
        f.array_sort(f.array_distinct(f.array_union(qc, f.array(f.lit(flag_text.value))))),
    ).otherwise(qc)
