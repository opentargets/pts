"""Schema collection to describe EuropePMC data."""

import polars as pl

epmc_schema = pl.Schema({
    'source': pl.String(),
    'firstPublicationDate': pl.String(),
    'pmid': pl.String(),
    'id': pl.String(),
    'pmcid': pl.String(),
})
