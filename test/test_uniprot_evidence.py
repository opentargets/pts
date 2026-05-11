"""Tests for the uniprot_evidence transformer."""

from __future__ import annotations

from pts.transformers.uniprot_evidence import _parse_record


def _lines(entry: str) -> list[str]:
    return entry.splitlines()


SINGLE_DISEASE_ENTRY = """\
ID   BRCA1_HUMAN              Reviewed;        1863 AA.
AC   P38398;
DE   RecName: Full=Breast cancer type 1 susceptibility protein;
GN   Name=BRCA1;
CC   -!- DISEASE: Breast-ovarian cancer, familial, 1 (BROVCA1) [MIM:604370]:
CC       A cancer predisposition syndrome with increased risk for breast
CC       and ovarian cancer. {ECO:0000269|PubMed:7545954,
CC       ECO:0000269|PubMed:9145676}.
"""


def test_parse_record_disease_omim():
    rec = _parse_record(_lines(SINGLE_DISEASE_ENTRY))
    assert len(rec['diseases']) == 1
    assert rec['diseases'][0]['omimId'] == '604370'


def test_parse_record_disease_name():
    rec = _parse_record(_lines(SINGLE_DISEASE_ENTRY))
    assert rec['diseases'][0]['name'] == 'Breast-ovarian cancer, familial, 1'


def test_parse_record_disease_acronym():
    rec = _parse_record(_lines(SINGLE_DISEASE_ENTRY))
    assert rec['diseases'][0]['acronym'] == 'BROVCA1'


def test_parse_record_disease_evidence_pmids():
    rec = _parse_record(_lines(SINGLE_DISEASE_ENTRY))
    assert rec['diseases'][0]['evidencePmids'] == ['7545954', '9145676']


DISEASE_WITH_COPYRIGHT_SEPARATOR = """\
ID   X_HUMAN                  Reviewed;         100 AA.
AC   Q11111;
CC   -!- DISEASE: A disease (XDIS) [MIM:111111]: text.
CC       {ECO:0000269|PubMed:1}.
CC   -----------------------------------------------------------------------
CC   Copyrighted by the UniProt Consortium.
"""


def test_parse_record_disease_not_polluted_by_copyright_block():
    rec = _parse_record(_lines(DISEASE_WITH_COPYRIGHT_SEPARATOR))
    assert len(rec['diseases']) == 1
    desc = rec['diseases'][0]['description']
    assert '---' not in desc
    assert 'Copyrighted' not in desc


def test_parse_record_gene_name_from_gn_line():
    rec = _parse_record(_lines(SINGLE_DISEASE_ENTRY))
    assert rec['geneNames'] == ['BRCA1']
