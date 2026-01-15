from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.expression_utils.run_cellex import CellexAnalysis


def baseline_expression_specificity(
    source: str,
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression specificity computation')

    # Initialize Spark Session
    if properties is None:
        properties = {}

    session = Session(app_name='baseline_expression_specificity', properties=properties)
    spark = session.spark

    # Determine input path
    input_path = source

    specificity_method = settings.get('specificity_method', 'cellex')

    if specificity_method == 'cellex':
        # Extract settings
        datasource = settings.get('datasource')
        biosample = settings.get('biosample')

        if not all([datasource, biosample]):
            logger.warning(
                 'Missing required settings for CELLEX: datasource or biosample. '
                 'Attempting to proceed but might fail if not inferred.'
             )

        cellex_analysis = CellexAnalysis(
            spark=spark,
            mode='parquet',
            input_path=input_path,
            output_path=destination,
            biosample=biosample,
            do_anova=settings.get('do_anova', True)
        )
        cellex_analysis.run()
    else:
        raise ValueError(f"Specificity method '{specificity_method}' not supported.")
