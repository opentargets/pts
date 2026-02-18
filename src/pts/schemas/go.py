"""Schema for Gene Ontology (GO) dataset."""

import polars as pl

go_schema = {
    'id': pl.String,
    'label': pl.String,
    'namespace': pl.String,
    'alt_ids': pl.List(pl.String),
    'is_a': pl.List(pl.String),
    'part_of': pl.List(pl.String),
    'regulates': pl.List(pl.String),
    'negatively_regulates': pl.List(pl.String),
    'positively_regulates': pl.List(pl.String),
    'isObsolete': pl.Boolean,
}
