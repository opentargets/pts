from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.expression_utils.dice import DiceBaselineExpression
from pts.pyspark.expression_utils.gtex import GtexBaselineExpression
from pts.pyspark.expression_utils.pride import PrideBaselineExpression
from pts.pyspark.expression_utils.pseudobulk_sc import PseudobulkExpression


def baseline_expression_harmonise(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression computation')

    # Initialize Spark Session
    if properties is None:
        properties = {}

    session = Session(app_name='baseline_expression_harmonise', properties=properties)
    spark = session.spark

    harmoniser = settings['harmoniser']
    if isinstance(harmoniser, list):
        harmoniser = harmoniser[0]

    if harmoniser == 'dice':
        # Extract arguments
        dice_directory = source['dice_directory']
        mapping_path = source['mapping_path']
        output_directory_path = destination['baseline_expression']

        # Run DICE processing
        DiceBaselineExpression(
            spark=spark,
            dice_directory=dice_directory,
            mapping_path=mapping_path,
            output_directory_path=output_directory_path,
            json=False,
            local=False
        ).run()
    elif harmoniser == 'tabula_sapiens':
        # Extract arguments
        h5ad_path = source['h5ad_path']
        output_directory_path = destination['baseline_expression']

        datasource_id = settings.get('datasource_id', 'tabula_sapiens')
        datatype_id = settings.get('datatype_id', 'scrna-seq')

        processor = PseudobulkExpression(
            spark=spark,
            h5ad_path=h5ad_path,
            output_directory_path=output_directory_path,
            datasource_id=datasource_id,
            datatype_id=datatype_id,
            json=False
        )

        processor.run(
            min_cells=settings.get('min_cells', 5),
            min_genes=settings.get('min_genes', 5),
            technology=settings.get('technology', '10X'),
            normalise=settings.get('normalise', False),
            aggregation_min_cells=settings.get('aggregation_min_cells', 5),
            aggregation_method=settings.get('aggregation_method', 'sum'),
            tissue_agg_colname=settings.get('tissue_agg_colname', 'tissue_ontology_term_id'),
            celltype_agg_colname=settings.get('celltype_agg_colname', 'cell_type_ontology_term_id'),
            age_colname=settings.get('age_colname', 'age'),
            sex_colname=settings.get('sex_colname', 'sex'),
            ethnicity_colname=settings.get('ethnicity_colname', 'ethnicity'),
            donor_colname=settings.get('donor_colname', 'donor_id')
        )

    elif harmoniser == 'pride':
        pride_source_data_dir = source['pride_source_data_dir']
        pride_codes = source['pride_codes']
        tissue_ontology_mapping_path = source['tissue_ontology_mapping_path']
        target_index_path = source['target_index_path']
        output_directory_path = destination['baseline_expression']

        PrideBaselineExpression(
            spark=spark,
            pride_source_data_dir=pride_source_data_dir,
            pride_codes=pride_codes,
            output_directory_path=output_directory_path,
            tissue_ontology_mapping_path=tissue_ontology_mapping_path,
            target_index_path=target_index_path,
            json=False,
            local=False
        ).run()

    elif harmoniser == 'gtex':
        gtex_source_data_path = source['gtex_source_data_path']
        sample_metadata_path = source['sample_metadata_path']
        subject_metadata_path = source['subject_metadata_path']
        output_directory_path = destination['baseline_expression']

        GtexBaselineExpression(
            spark=spark,
            gtex_source_data_path=gtex_source_data_path,
            output_directory_path=output_directory_path,
            sample_metadata_path=sample_metadata_path,
            subject_metadata_path=subject_metadata_path,
            json=False,
            local=False,
            matrix=settings.get('matrix', False)
        ).run()

    else:
        logger.warning(f'Harmoniser {harmoniser} not implemented yet')
