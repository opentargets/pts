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


MULTI_DISEASE_ENTRY = """\
ID   TP53_HUMAN               Reviewed;         393 AA.
AC   P04637;
GN   Name=TP53;
CC   -!- DISEASE: Li-Fraumeni syndrome 1 (LFS1) [MIM:151623]: Autosomal
CC       dominant. {ECO:0000269|PubMed:1565144}.
CC   -!- DISEASE: Esophageal cancer (ESCR) [MIM:133239]: Malignancy.
CC       {ECO:0000269|PubMed:8632902, ECO:0000269|PubMed:10780666}.
CC   -!- FUNCTION: Acts as a tumor suppressor.
"""


def test_parse_record_multi_disease_count():
    rec = _parse_record(_lines(MULTI_DISEASE_ENTRY))
    assert len(rec['diseases']) == 2


def test_parse_record_multi_disease_acronyms():
    rec = _parse_record(_lines(MULTI_DISEASE_ENTRY))
    acronyms = [d['acronym'] for d in rec['diseases']]
    assert acronyms == ['LFS1', 'ESCR']


def test_parse_record_disease_block_terminated_by_non_disease_cc():
    rec = _parse_record(_lines(MULTI_DISEASE_ENTRY))
    second = rec['diseases'][1]
    assert second['evidencePmids'] == ['8632902', '10780666']
    assert 'tumor suppressor' not in second['description']


def test_parse_record_no_diseases():
    entry = """\
ID   NODIS_HUMAN              Reviewed;         100 AA.
AC   Q12345;
GN   Name=NODIS;
CC   -!- FUNCTION: Some function.
"""
    rec = _parse_record(_lines(entry))
    assert rec['diseases'] == []


def test_parse_record_id_and_accession_and_gene():
    rec = _parse_record(_lines(MULTI_DISEASE_ENTRY))
    assert rec['id'] == 'TP53_HUMAN'
    assert rec['accession'] == 'P04637'
    assert rec['geneNames'] == ['TP53']
