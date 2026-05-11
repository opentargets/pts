"""Shared helpers for the uniprot_variants and uniprot_literature evidence tasks."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as f

DATASOURCE_VARIANTS = 'uniprot_variants'
DATASOURCE_LITERATURE = 'uniprot_literature'

DATATYPE_GENETIC_ASSOCIATION = 'genetic_association'
DATATYPE_GENETIC_LITERATURE = 'genetic_literature'

TARGET_MODULATION = 'up_or_down'

# Phrases that, when present in a CC DISEASE block's description, indicate
# UniProt curators have flagged the gene-disease association as "indefinite"
# (likely-not-causative). Ported from the legacy Java pipeline's
# DefaultBaseFactory.INDEFINITE_DISEASE_NOTE_ASSOCIATIONS, with `mutations` /
# `variations` updated to `variants` to match modern UniProt phrasing —
# the legacy strings produce a near-zero match rate against current data.
# Rows matching any of these phrases get confidence='medium'; all others get 'high'.
INDEFINITE_DISEASE_NOTES = (
    'The disease may be caused by variants affecting the gene represented in this entry',
    'The disease may be caused by variants affecting distinct genetic loci, including the gene represented in this entry',  # noqa: E501
    'Disease susceptibility may be associated with variants affecting the gene represented in this entry',
    'The gene represented in this entry may act as a disease modifier',
    'The gene represented in this entry may be involved in disease pathogenesis',
    'The protein represented in this entry may be involved in disease pathogenesis',
)


def confidence_from_description(description_col: Column) -> Column:
    """Return 'medium' if the disease description contains any indefinite-association phrase, 'high' otherwise.

    The score_expression in evidence_postprocess maps high -> 1.0, medium -> 0.5.
    """
    matches_indefinite: Column | None = None
    for phrase in INDEFINITE_DISEASE_NOTES:
        cond = description_col.contains(phrase)
        matches_indefinite = cond if matches_indefinite is None else (matches_indefinite | cond)
    return f.when(matches_indefinite, f.lit('medium')).otherwise(f.lit('high'))
