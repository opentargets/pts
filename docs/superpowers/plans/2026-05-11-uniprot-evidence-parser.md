# UniProt Evidence Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Java `uniprot-evidence-parser` to native PTS as three new steps (`uniprot_evidence_parse`, `uniprot_variants`, `uniprot_literature`) that emit parquet evidence rows consumable by the existing `evidence_postprocess_uniprot_*` steps.

**Architecture:** A Polars transformer parses the Swiss-Prot flat file once into an intermediate parquet (`intermediate/evidence/uniprot/uniprot_evidence.parquet`) carrying disease comments and variant features per entry. Two PySpark tasks consume that intermediate, join the ontoma EFO LUTs via `add_efo_mapping`, and emit `intermediate/evidence/uniprot_variants.parquet` and `intermediate/evidence/uniprot_literature.parquet`. The somatic-vs-germline distinction for variants is sourced from a checked-in `uniprot_somatic_census.txt` static input (ported verbatim from the Java repo — see Task 8 for the rationale this overrides the design's preferred option 2).

**Tech Stack:** Python 3.13, Polars (parser), PySpark (evidence tasks), Otter (orchestration), pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-05-11-uniprot-evidence-parser-design.md`](../specs/2026-05-11-uniprot-evidence-parser-design.md)

---

## File map

| Path | Status | Responsibility |
|---|---|---|
| `src/pts/transformers/uniprot_evidence.py` | Create | Polars parser: Swiss-Prot flat file → intermediate parquet |
| `test/test_uniprot_evidence.py` | Create | Unit tests for `_parse_record` and helpers |
| `src/pts/pyspark/evidence_utils/uniprot.py` | Create | URL builder, confidence rule, census loader |
| `src/pts/pyspark/uniprot_literature.py` | Create | Literature evidence PySpark task |
| `test/test_uniprot_literature.py` | Create | Unit tests for the literature task |
| `src/pts/pyspark/uniprot_variants.py` | Create | Variants evidence PySpark task |
| `test/test_uniprot_variants.py` | Create | Unit tests for the variants task |
| `input/evidence/uniprot/uniprot_somatic_census.txt` | Create (committed) | Static somatic dbSNP rsID census ported from Java repo |
| `config.yaml` | Modify | Register three new steps + retarget two `evidence_postprocess_uniprot_*` |

---

## Task 1: Parser — CC DISEASE block

**Files:**
- Create: `src/pts/transformers/uniprot_evidence.py`
- Create: `test/test_uniprot_evidence.py`

- [ ] **Step 1.1: Write the failing test**

Create `test/test_uniprot_evidence.py` with:

```python
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
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: `ModuleNotFoundError: No module named 'pts.transformers.uniprot_evidence'`

- [ ] **Step 1.3: Implement minimal parser to make the four tests pass**

Create `src/pts/transformers/uniprot_evidence.py`:

```python
"""Transformer that parses Swiss-Prot flat files into an evidence-oriented parquet.

Output is consumed by the `uniprot_variants` and `uniprot_literature` PySpark tasks.
"""

from __future__ import annotations

import re

_BRACES_RE = re.compile(r'\{[^}]*\}')
_ECO_PUBMED_RE = re.compile(r'ECO:\d+\|PubMed:(\d+)')
_DISEASE_HEADER_RE = re.compile(
    r'^(?P<name>.+?)\s*\((?P<acronym>[^)]+)\)\s*\[MIM:(?P<omim>\d+)\]:\s*(?P<rest>.*)$'
)


def _strip_braces(text: str) -> str:
    return _BRACES_RE.sub('', text).strip()


def _parse_disease_block(text: str) -> dict | None:
    """Parse a single concatenated CC DISEASE block text.

    The input is the joined content lines for one disease (without `CC` prefixes),
    including the `{ECO:...}` evidence segments still intact.
    """
    pmids = _ECO_PUBMED_RE.findall(text)
    cleaned = _strip_braces(text)
    m = _DISEASE_HEADER_RE.match(cleaned)
    if not m:
        return None
    return {
        'omimId': m.group('omim'),
        'name': m.group('name').strip(),
        'acronym': m.group('acronym').strip(),
        'description': m.group('rest').strip().rstrip('.').strip(),
        'evidencePmids': pmids,
    }


def _parse_record(lines: list[str]) -> dict:
    entry_id = ''
    accession = ''
    gene_names: list[str] = []
    diseases: list[dict] = []
    variants: list[dict] = []

    # CC DISEASE accumulator
    in_disease = False
    disease_lines: list[str] = []

    def _flush_disease() -> None:
        nonlocal in_disease, disease_lines
        if disease_lines:
            text = ' '.join(disease_lines).strip()
            parsed = _parse_disease_block(text)
            if parsed is not None:
                diseases.append(parsed)
        in_disease = False
        disease_lines = []

    for raw in lines:
        if len(raw) < 2:
            continue
        code = raw[:2]
        content = raw[5:] if len(raw) > 5 else ''

        if code == 'ID':
            tokens = content.split()
            if tokens:
                entry_id = tokens[0]

        elif code == 'AC':
            if not accession:
                cleaned = _strip_braces(content)
                for ac in cleaned.split(';'):
                    ac = ac.strip()
                    if ac:
                        accession = ac
                        break

        elif code == 'GN':
            cleaned = _strip_braces(content)
            for segment in cleaned.split(';'):
                segment = segment.strip()
                if '=' in segment:
                    key, _, value = segment.partition('=')
                    if key.strip() == 'Name':
                        for v in value.split(','):
                            v = v.strip().rstrip(';')
                            if v:
                                gene_names.append(v)

        elif code == 'CC':
            stripped_content = content.lstrip()
            if stripped_content.startswith('-!- DISEASE:'):
                _flush_disease()
                in_disease = True
                first = stripped_content[len('-!- DISEASE:'):].strip()
                if first:
                    disease_lines.append(first)
            elif stripped_content.startswith('-!- '):
                _flush_disease()
            elif in_disease:
                cont = content.strip()
                if cont:
                    disease_lines.append(cont)

    _flush_disease()

    return {
        'id': entry_id,
        'accession': accession,
        'geneNames': gene_names,
        'diseases': diseases,
        'variants': variants,
    }
```

- [ ] **Step 1.4: Run the tests to verify they pass**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: all four tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/pts/transformers/uniprot_evidence.py test/test_uniprot_evidence.py
git commit -m "feat(uniprot-evidence): parser scaffolding with CC DISEASE block extraction"
```

---

## Task 2: Parser — multi-disease entries and edge cases

**Files:**
- Modify: `test/test_uniprot_evidence.py` (add tests)
- Modify: `src/pts/transformers/uniprot_evidence.py` (no logic change expected; verify generality)

- [ ] **Step 2.1: Write the failing tests**

Append to `test/test_uniprot_evidence.py`:

```python
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
```

- [ ] **Step 2.2: Run the tests**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: all five new tests PASS (Task 1's `_flush_disease` and CC state machine already handle these).

- [ ] **Step 2.3: Commit**

```bash
git add test/test_uniprot_evidence.py
git commit -m "test(uniprot-evidence): multi-disease and edge-case coverage"
```

---

## Task 3: Parser — FT VARIANT features

**Files:**
- Modify: `test/test_uniprot_evidence.py`
- Modify: `src/pts/transformers/uniprot_evidence.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `test/test_uniprot_evidence.py`:

```python
SINGLE_VARIANT_ENTRY = """\
ID   BRCA1_HUMAN              Reviewed;        1863 AA.
AC   P38398;
GN   Name=BRCA1;
CC   -!- DISEASE: Breast-ovarian cancer, familial, 1 (BROVCA1) [MIM:604370]:
CC       A cancer. {ECO:0000269|PubMed:1111111}.
FT   VARIANT         1699
FT                   /note="R -> Q (in BROVCA1; dbSNP:rs28897696)"
FT                   /evidence="ECO:0000269|PubMed:9145676"
FT                   /id="VAR_007800"
FT                   /db_snp="rs28897696"
"""


def test_parse_record_variant_count():
    rec = _parse_record(_lines(SINGLE_VARIANT_ENTRY))
    assert len(rec['variants']) == 1


def test_parse_record_variant_ft_id():
    rec = _parse_record(_lines(SINGLE_VARIANT_ENTRY))
    assert rec['variants'][0]['ftId'] == 'VAR_007800'


def test_parse_record_variant_dbsnp():
    rec = _parse_record(_lines(SINGLE_VARIANT_ENTRY))
    assert rec['variants'][0]['dbSnpRsId'] == 'rs28897696'


def test_parse_record_variant_aa_change():
    rec = _parse_record(_lines(SINGLE_VARIANT_ENTRY))
    assert rec['variants'][0]['aminoacidChange'] == 'p.Arg1699Gln'


def test_parse_record_variant_evidence_pmids():
    rec = _parse_record(_lines(SINGLE_VARIANT_ENTRY))
    assert rec['variants'][0]['evidencePmids'] == ['9145676']


def test_parse_record_variant_description_in_phrase():
    rec = _parse_record(_lines(SINGLE_VARIANT_ENTRY))
    assert 'BROVCA1' in rec['variants'][0]['description']


def test_parse_record_variant_no_dbsnp():
    entry = """\
ID   X_HUMAN                  Reviewed;         100 AA.
AC   Q99999;
FT   VARIANT         50
FT                   /note="A -> V (in some disease)"
FT                   /id="VAR_900000"
"""
    rec = _parse_record(_lines(entry))
    assert rec['variants'][0]['dbSnpRsId'] is None
```

- [ ] **Step 3.2: Run the tests to verify they fail**

Run: `uv run pytest test/test_uniprot_evidence.py -v -k variant`
Expected: all seven new tests FAIL with `assert ... == 1` (variants list is empty) or KeyError.

- [ ] **Step 3.3: Implement FT VARIANT parsing**

In `src/pts/transformers/uniprot_evidence.py`:

Add these regexes at module level after `_DISEASE_HEADER_RE`:

```python
_VARIANT_POS_RE = re.compile(r'^VARIANT\s+(?P<pos>\d+)(?:\s*\.\.\s*\d+)?\s*$')
_VARIANT_NOTE_RE = re.compile(r'^/note="(?P<change>[^"]+?)\s*(?:\((?P<rest>.*)\))?"$', re.DOTALL)
_VARIANT_NOTE_OPEN_RE = re.compile(r'^/note="(.*)$')
_VARIANT_ID_RE = re.compile(r'^/id="(?P<id>[^"]+)"$')
_VARIANT_DBSNP_RE = re.compile(r'^/db_snp="(?P<rsid>rs\d+)"$')
_VARIANT_EVIDENCE_RE = re.compile(r'^/evidence="(?P<value>[^"]+)"$')
_VARIANT_EVIDENCE_OPEN_RE = re.compile(r'^/evidence="(.*)$')
_AA_CHANGE_RE = re.compile(r'^(?P<from>[A-Z])\s*->\s*(?P<to>[A-Z])$')

_AA_THREE_LETTER = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
}


def _format_aa_change(position: str, change_text: str) -> str:
    m = _AA_CHANGE_RE.match(change_text.strip())
    if not m:
        return ''
    one_from = _AA_THREE_LETTER.get(m.group('from'), '')
    one_to = _AA_THREE_LETTER.get(m.group('to'), '')
    if not one_from or not one_to:
        return ''
    return f'p.{one_from}{position}{one_to}'


def _parse_variant_qualifiers(position: str, qualifier_text: str) -> dict | None:
    """Parse the concatenated /note=... /id=... /db_snp=... block.

    `qualifier_text` is the joined continuation content (FT lines with the leading
    `FT   ` stripped). Returns a variant dict or None if no `/id` was found.
    """
    qualifiers: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []
    accumulated = qualifier_text.strip()
    # Re-split on a leading `/` boundary while respecting quoted values that
    # may contain `/`. UniProt qualifier blocks always begin a new qualifier
    # with `/key="`. We split on that pattern.
    parts = re.split(r'(?=(?<![A-Za-z0-9_])/[a-z_]+=")', accumulated)
    for part in parts:
        part = part.strip()
        if not part.startswith('/'):
            continue
        key_match = re.match(r'^/([a-z_]+)="(.*)"\s*$', part, re.DOTALL)
        if key_match:
            qualifiers[key_match.group(1)] = key_match.group(2)

    note = qualifiers.get('note', '')
    ft_id = qualifiers.get('id', '')
    if not ft_id:
        return None

    db_snp = qualifiers.get('db_snp')
    evidence_text = qualifiers.get('evidence', '')
    evidence_pmids = _ECO_PUBMED_RE.findall(evidence_text)

    # Parse change + parenthesised description from note
    change_text = note
    description = ''
    if '(' in note and note.endswith(')'):
        idx = note.index('(')
        change_text = note[:idx].strip()
        description = note[idx + 1:-1].strip()
    elif '(' in note:
        idx = note.index('(')
        change_text = note[:idx].strip()
        description = note[idx + 1:].strip()

    aa_change = _format_aa_change(position, change_text)

    return {
        'ftId': ft_id,
        'description': description or note.strip(),
        'aminoacidChange': aa_change,
        'dbSnpRsId': db_snp,
        'linkedOmimIds': [],
        'evidencePmids': evidence_pmids,
    }
```

Then update `_parse_record` to consume `FT` lines. Add this state-machine logic inside the for-loop, after the `CC` branch:

```python
        elif code == 'FT':
            ft_content = raw[5:].rstrip()  # preserve internal spacing
            # New feature line: starts at column 5 (no leading spaces in content)
            if ft_content and not ft_content.startswith(' '):
                # Flush previous variant if any
                if ft_in_variant:
                    parsed = _parse_variant_qualifiers(ft_position, ' '.join(ft_qualifier_lines))
                    if parsed is not None:
                        variants.append(parsed)
                    ft_in_variant = False
                    ft_qualifier_lines = []
                    ft_position = ''
                pos_match = _VARIANT_POS_RE.match(ft_content)
                if pos_match:
                    ft_in_variant = True
                    ft_position = pos_match.group('pos')
                    ft_qualifier_lines = []
            elif ft_in_variant:
                # Continuation line
                ft_qualifier_lines.append(ft_content.strip())
```

And declare the FT state above the for-loop, alongside the disease state:

```python
    ft_in_variant = False
    ft_position = ''
    ft_qualifier_lines: list[str] = []
```

And flush the FT state after the for-loop, after `_flush_disease()`:

```python
    if ft_in_variant:
        parsed = _parse_variant_qualifiers(ft_position, ' '.join(ft_qualifier_lines))
        if parsed is not None:
            variants.append(parsed)
```

- [ ] **Step 3.4: Run the tests to verify they pass**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: all tests PASS (disease tests still green, all seven variant tests green).

- [ ] **Step 3.5: Commit**

```bash
git add src/pts/transformers/uniprot_evidence.py test/test_uniprot_evidence.py
git commit -m "feat(uniprot-evidence): FT VARIANT feature extraction"
```

---

## Task 4: Parser — variant ↔ disease linkage by acronym

**Files:**
- Modify: `test/test_uniprot_evidence.py`
- Modify: `src/pts/transformers/uniprot_evidence.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `test/test_uniprot_evidence.py`:

```python
LINKED_VARIANT_ENTRY = """\
ID   BRCA1_HUMAN              Reviewed;        1863 AA.
AC   P38398;
GN   Name=BRCA1;
CC   -!- DISEASE: Breast-ovarian cancer, familial, 1 (BROVCA1) [MIM:604370]:
CC       A cancer. {ECO:0000269|PubMed:1111111}.
CC   -!- DISEASE: Pancreatic cancer 4 (PNCA4) [MIM:614320]:
CC       Another condition. {ECO:0000269|PubMed:2222222}.
FT   VARIANT         1699
FT                   /note="R -> Q (in BROVCA1; dbSNP:rs28897696)"
FT                   /evidence="ECO:0000269|PubMed:9145676"
FT                   /id="VAR_007800"
FT                   /db_snp="rs28897696"
FT   VARIANT         1738
FT                   /note="C -> Y (unknown significance)"
FT                   /id="VAR_007801"
"""


def test_variant_linked_to_disease_by_acronym():
    rec = _parse_record(_lines(LINKED_VARIANT_ENTRY))
    by_id = {v['ftId']: v for v in rec['variants']}
    assert by_id['VAR_007800']['linkedOmimIds'] == ['604370']


def test_variant_unlinked_when_no_acronym_match():
    rec = _parse_record(_lines(LINKED_VARIANT_ENTRY))
    by_id = {v['ftId']: v for v in rec['variants']}
    assert by_id['VAR_007801']['linkedOmimIds'] == []
```

- [ ] **Step 4.2: Run the tests to verify they fail**

Run: `uv run pytest test/test_uniprot_evidence.py -v -k linked`
Expected: `test_variant_linked_to_disease_by_acronym` FAILS (linkedOmimIds is empty), `test_variant_unlinked_when_no_acronym_match` PASSES (already empty).

- [ ] **Step 4.3: Implement linkage**

In `src/pts/transformers/uniprot_evidence.py`, add a helper above `_parse_record`:

```python
def _link_variants_to_diseases(diseases: list[dict], variants: list[dict]) -> None:
    """Attach `linkedOmimIds` to each variant by matching disease acronyms in its description.

    Mutates `variants` in place. Matching is whole-word and case-sensitive on the
    acronym (UniProt acronyms are uppercase short identifiers like BROVCA1, LFS1).
    """
    if not diseases or not variants:
        return
    acronym_to_omim = {d['acronym']: d['omimId'] for d in diseases if d.get('acronym')}
    if not acronym_to_omim:
        return
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(a) for a in acronym_to_omim) + r')\b'
    )
    for variant in variants:
        matches = pattern.findall(variant['description'])
        seen: set[str] = set()
        linked: list[str] = []
        for acronym in matches:
            omim = acronym_to_omim[acronym]
            if omim not in seen:
                seen.add(omim)
                linked.append(omim)
        variant['linkedOmimIds'] = linked
```

Then call it inside `_parse_record` immediately before the `return` statement:

```python
    _link_variants_to_diseases(diseases, variants)
```

- [ ] **Step 4.4: Run the tests**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: all tests PASS, including both linkage tests.

- [ ] **Step 4.5: Commit**

```bash
git add src/pts/transformers/uniprot_evidence.py test/test_uniprot_evidence.py
git commit -m "feat(uniprot-evidence): link variants to diseases by acronym"
```

---

## Task 5: Parser — streaming over `//`-delimited entries

**Files:**
- Modify: `test/test_uniprot_evidence.py`
- Modify: `src/pts/transformers/uniprot_evidence.py`

- [ ] **Step 5.1: Write the failing test**

Append to `test/test_uniprot_evidence.py`:

```python
import io

from pts.transformers.uniprot_evidence import _parse_uniprot


def test_parse_uniprot_streams_multiple_records():
    flat = """\
ID   A_HUMAN                  Reviewed;         100 AA.
AC   Q11111;
GN   Name=A;
CC   -!- DISEASE: A disease, familial (ADIS) [MIM:111111]: text.
CC       {ECO:0000269|PubMed:1}.
//
ID   B_HUMAN                  Reviewed;         100 AA.
AC   Q22222;
GN   Name=B;
CC   -!- DISEASE: B disease (BDIS) [MIM:222222]: text.
CC       {ECO:0000269|PubMed:2}.
//
"""
    records = _parse_uniprot(io.StringIO(flat))
    assert [r['accession'] for r in records] == ['Q11111', 'Q22222']
    assert records[0]['diseases'][0]['omimId'] == '111111'
    assert records[1]['diseases'][0]['omimId'] == '222222'
```

- [ ] **Step 5.2: Run the test to verify it fails**

Run: `uv run pytest test/test_uniprot_evidence.py::test_parse_uniprot_streams_multiple_records -v`
Expected: `ImportError: cannot import name '_parse_uniprot'`.

- [ ] **Step 5.3: Implement `_parse_uniprot`**

Append to `src/pts/transformers/uniprot_evidence.py`:

```python
from loguru import logger


def _parse_uniprot(file_obj) -> list[dict]:
    records: list[dict] = []
    current_lines: list[str] = []
    count = 0

    for raw_line in file_obj:
        if isinstance(raw_line, bytes):
            line = raw_line.decode('utf-8', errors='replace').rstrip('\n')
        else:
            line = raw_line.rstrip('\n')

        if line.startswith('//'):
            if current_lines:
                records.append(_parse_record(current_lines))
                count += 1
                if count % 10000 == 0:
                    logger.info(f'parsed {count} uniprot records')
                current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        records.append(_parse_record(current_lines))

    logger.info(f'parsed {len(records)} uniprot records total')
    return records
```

(The `loguru` import goes at the top of the file with the existing imports.)

- [ ] **Step 5.4: Run the tests**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: all tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/pts/transformers/uniprot_evidence.py test/test_uniprot_evidence.py
git commit -m "feat(uniprot-evidence): stream entries delimited by //"
```

---

## Task 6: Public transformer entry point

**Files:**
- Modify: `src/pts/transformers/uniprot_evidence.py`
- Modify: `test/test_uniprot_evidence.py`

- [ ] **Step 6.1: Write the failing test**

Append to `test/test_uniprot_evidence.py`:

```python
import gzip
from pathlib import Path

import polars as pl

from pts.transformers.uniprot_evidence import uniprot_evidence


def test_uniprot_evidence_writes_parquet(tmp_path):
    src_path = tmp_path / 'mini.txt.gz'
    payload = b"""\
ID   A_HUMAN                  Reviewed;         100 AA.
AC   Q11111;
GN   Name=A;
CC   -!- DISEASE: A disease (ADIS) [MIM:111111]: text.
CC       {ECO:0000269|PubMed:1}.
FT   VARIANT         42
FT                   /note="R -> Q (in ADIS)"
FT                   /id="VAR_000001"
FT                   /db_snp="rs1"
//
"""
    with gzip.open(src_path, 'wb') as fh:
        fh.write(payload)

    dst_path = tmp_path / 'out.parquet'
    uniprot_evidence(str(src_path), str(dst_path), {}, config=None)

    df = pl.read_parquet(str(dst_path))
    assert df.height == 1
    row = df.row(0, named=True)
    assert row['accession'] == 'Q11111'
    assert row['diseases'][0]['omimId'] == '111111'
    assert row['variants'][0]['linkedOmimIds'] == ['111111']
```

- [ ] **Step 6.2: Run the test to verify it fails**

Run: `uv run pytest test/test_uniprot_evidence.py::test_uniprot_evidence_writes_parquet -v`
Expected: `ImportError: cannot import name 'uniprot_evidence'`.

- [ ] **Step 6.3: Implement `uniprot_evidence()` and the Polars schema**

Append to `src/pts/transformers/uniprot_evidence.py`:

```python
import gzip
from typing import Any

import polars as pl
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle


_UNIPROT_EVIDENCE_SCHEMA = pl.Schema({
    'id': pl.String,
    'accession': pl.String,
    'geneNames': pl.List(pl.String),
    'diseases': pl.List(pl.Struct({
        'omimId': pl.String,
        'name': pl.String,
        'acronym': pl.String,
        'description': pl.String,
        'evidencePmids': pl.List(pl.String),
    })),
    'variants': pl.List(pl.Struct({
        'ftId': pl.String,
        'description': pl.String,
        'aminoacidChange': pl.String,
        'dbSnpRsId': pl.String,
        'linkedOmimIds': pl.List(pl.String),
        'evidencePmids': pl.List(pl.String),
    })),
})


def uniprot_evidence(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    logger.info(f'loading uniprot flat file from {source}')
    h = StorageHandle(source)

    with h.open('rb') as raw:
        if source.endswith('.gz'):
            with gzip.open(raw, 'rb') as gz:
                rows = _parse_uniprot(gz)
        else:
            rows = _parse_uniprot(raw)

    logger.info('creating evidence dataframe')
    df = pl.DataFrame(rows, schema=_UNIPROT_EVIDENCE_SCHEMA)

    logger.info(f'writing uniprot evidence parquet to {destination}')
    df.write_parquet(destination)
```

(Move the `loguru` import up to consolidate with the new imports at the top if useful, but it can stay where it is.)

- [ ] **Step 6.4: Run the tests**

Run: `uv run pytest test/test_uniprot_evidence.py -v`
Expected: all tests PASS, including the new parquet round-trip test.

- [ ] **Step 6.5: Commit**

```bash
git add src/pts/transformers/uniprot_evidence.py test/test_uniprot_evidence.py
git commit -m "feat(uniprot-evidence): public transformer entry writing parquet"
```

---

## Task 7: Wire the parser into config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 7.1: Add the new step**

Locate the comment block that ends with the existing `transform uniprot` task (around line 226 — the `## TARGET STEP (pyspark)` separator). Insert a new top-level step **before** the `target:` step:

```yaml
  #: UNIPROT EVIDENCE PARSE STEP :##################################################################
  uniprot_evidence_parse:
    - name: transform uniprot evidence
      transformer: uniprot_evidence
      source: input/target/uniprot/uniprot.txt.gz
      destination: intermediate/evidence/uniprot/uniprot_evidence.parquet
  ##################################################################################################
```

- [ ] **Step 7.2: Verify config parses**

Run: `uv run pts --step uniprot_evidence_parse --help` or `uv run pts -h`
Expected: command exits 0 (no YAML parse error). The step name `uniprot_evidence_parse` is now listed.

- [ ] **Step 7.3: Commit**

```bash
git add config.yaml
git commit -m "feat(config): register uniprot_evidence_parse step"
```

---

## Task 8: Port the static somatic census file

**Note on the choice of input:** The design proposed humsavar.txt as the primary somatic source (option 2). On investigation, humsavar's `CATEGORY` column carries clinical-significance values (LP / LB / US) — likelihood of pathogenicity, not somatic-vs-germline origin. There's no column in humsavar that distinguishes somatic from germline. The design's documented fallback (option 1) is therefore the working approach: port the curated `uniprot_somatic_census.txt` from the legacy Java repo as a static input. If you later identify a separate UniProt resource that exposes somatic-vs-germline origin per rsID, this can be revisited.

**Files:**
- Create: `input/evidence/uniprot/uniprot_somatic_census.txt`

- [ ] **Step 8.1: Copy the file from the Java repo**

Run:

```bash
mkdir -p input/evidence/uniprot
cp /Users/ochoa/Projects/uniprot-evidence-parser/src/bin/uniprot_somatic_census.txt \
   input/evidence/uniprot/uniprot_somatic_census.txt
```

- [ ] **Step 8.2: Verify it landed**

Run:

```bash
wc -l input/evidence/uniprot/uniprot_somatic_census.txt
head -5 input/evidence/uniprot/uniprot_somatic_census.txt
```

Expected: a non-zero line count and the first few lines visible (rsID-like tokens).

- [ ] **Step 8.3: Commit**

```bash
git add input/evidence/uniprot/uniprot_somatic_census.txt
git commit -m "chore(uniprot-evidence): port somatic dbSNP census from Java repo"
```

---

## Task 9: Shared `evidence_utils/uniprot.py`

**Files:**
- Create: `src/pts/pyspark/evidence_utils/uniprot.py`

- [ ] **Step 9.1: Write the module**

```python
"""Shared helpers for the uniprot_variants and uniprot_literature evidence tasks."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f

UNIPROT_BASE_URL = 'https://www.uniprot.org/uniprotkb/'

DATASOURCE_VARIANTS = 'uniprot_variants'
DATASOURCE_LITERATURE = 'uniprot_literature'

DATATYPE_GENETIC_ASSOCIATION = 'genetic_association'
DATATYPE_SOMATIC_MUTATIONS = 'somatic_mutations'


def uniprot_url(accession_col: Column) -> Column:
    """Build the canonical UniProt URL for an accession column."""
    return f.concat(f.lit(UNIPROT_BASE_URL), accession_col)


def uniprot_urls_struct_array(accession_col: Column) -> Column:
    """Return an array(struct(niceName, url)) shaped like other PTS evidence pipelines."""
    return f.array(
        f.struct(
            f.lit('UniProt').alias('niceName'),
            uniprot_url(accession_col).alias('url'),
        )
    )


def confidence_from_literature(literature_col: Column) -> Column:
    """High when at least one citing PMID is present, medium otherwise.

    Used to drive the score_expression mapping in evidence_postprocess
    (high -> 1.0, medium -> 0.5).
    """
    return f.when(f.size(literature_col) > 0, f.lit('high')).otherwise(f.lit('medium'))


def load_somatic_rsids(spark: SparkSession, path: str) -> DataFrame:
    """Read the static somatic dbSNP census into a single-column DataFrame.

    Each non-empty, non-comment line is treated as a single rsID. Returns a DataFrame
    with one column `dbSnpRsId` of unique rsIDs, suitable for a broadcast join.
    """
    raw = spark.read.text(path).withColumnRenamed('value', 'line')
    return (
        raw.select(f.trim(f.col('line')).alias('line'))
        .filter((f.length('line') > 0) & (~f.col('line').startswith('#')))
        .select(f.col('line').alias('dbSnpRsId'))
        .distinct()
    )
```

- [ ] **Step 9.2: Smoke-import in a quick session**

Run: `uv run python -c "from pts.pyspark.evidence_utils.uniprot import uniprot_url, confidence_from_literature, load_somatic_rsids; print('ok')"`
Expected: `ok`.

- [ ] **Step 9.3: Commit**

```bash
git add src/pts/pyspark/evidence_utils/uniprot.py
git commit -m "feat(uniprot-evidence): shared helpers for url, confidence, somatic census"
```

---

## Task 10: `uniprot_literature` PySpark task

**Files:**
- Create: `src/pts/pyspark/uniprot_literature.py`
- Create: `test/test_uniprot_literature.py`

- [ ] **Step 10.1: Write the failing test**

Create `test/test_uniprot_literature.py`:

```python
"""Tests for the uniprot_literature pyspark task."""

from __future__ import annotations

from pyspark.sql import Row


def _make_parsed_row(
    accession='P38398',
    diseases=None,
    variants=None,
):
    return Row(
        id='BRCA1_HUMAN',
        accession=accession,
        geneNames=['BRCA1'],
        diseases=diseases or [],
        variants=variants or [],
    )


def test_uniprot_literature_projects_expected_columns(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_literature as mod

    parsed = [
        _make_parsed_row(
            diseases=[
                Row(
                    omimId='604370',
                    name='Breast-ovarian cancer 1',
                    acronym='BROVCA1',
                    description='Cancer.',
                    evidencePmids=['7545954', '9145676'],
                ),
                Row(
                    omimId='999999',
                    name='Empty-evidence disease',
                    acronym='EMPTY',
                    description='No PMIDs.',
                    evidencePmids=[],
                ),
            ],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed).write.parquet(str(parsed_path))

    # Stub add_efo_mapping: behave as a no-op that adds a null column
    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'literature.parquet'
    mod._compute_literature(
        spark=spark,
        parsed_path=str(parsed_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    rows = df.collect()

    assert len(rows) == 1  # the empty-evidence disease is filtered out
    r = rows[0]
    assert r['datasourceId'] == 'uniprot_literature'
    assert r['datatypeId'] == 'genetic_association'
    assert r['targetFromSourceId'] == 'P38398'
    assert r['diseaseFromSource'] == 'Breast-ovarian cancer 1'
    assert r['diseaseFromSourceId'] == 'OMIM:604370'
    assert r['literature'] == ['7545954', '9145676']
    assert r['confidence'] == 'high'
    assert r['urls'][0]['niceName'] == 'UniProt'
    assert r['urls'][0]['url'] == 'https://www.uniprot.org/uniprotkb/P38398'
```

- [ ] **Step 10.2: Run the test to verify it fails**

Run: `uv run pytest test/test_uniprot_literature.py -v`
Expected: `ModuleNotFoundError: No module named 'pts.pyspark.uniprot_literature'`.

- [ ] **Step 10.3: Implement the task**

Create `src/pts/pyspark/uniprot_literature.py`:

```python
"""Evidence parser for UniProt disease-association literature curation."""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session
from pts.pyspark.evidence_utils.uniprot import (
    DATASOURCE_LITERATURE,
    DATATYPE_GENETIC_ASSOCIATION,
    confidence_from_literature,
    uniprot_urls_struct_array,
)


def _compute_literature(
    spark: SparkSession,
    parsed_path: str,
    disease_label_lut_path: str,
    disease_id_lut_path: str,
) -> DataFrame:
    parsed = spark.read.parquet(parsed_path)
    exploded = parsed.select(
        f.col('accession'),
        f.explode('diseases').alias('disease'),
    )

    with_literature = exploded.filter(f.size(f.col('disease.evidencePmids')) > 0)

    projected = with_literature.select(
        f.lit(DATASOURCE_LITERATURE).alias('datasourceId'),
        f.lit(DATATYPE_GENETIC_ASSOCIATION).alias('datatypeId'),
        f.col('accession').alias('targetFromSourceId'),
        f.col('disease.name').alias('diseaseFromSource'),
        f.concat(f.lit('OMIM:'), f.col('disease.omimId')).alias('diseaseFromSourceId'),
        f.col('disease.evidencePmids').alias('literature'),
        confidence_from_literature(f.col('disease.evidencePmids')).alias('confidence'),
        uniprot_urls_struct_array(f.col('accession')).alias('urls'),
    )

    logger.info('map uniprot literature diseases to EFO')
    mapped = add_efo_mapping(
        spark=spark,
        evidence_df=projected,
        disease_label_lut_path=disease_label_lut_path,
        disease_id_lut_path=disease_id_lut_path,
    )
    return mapped


def uniprot_literature(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='uniprot_literature', properties=properties)
    logger.info(f'load data from {source}')

    result_df = _compute_literature(
        spark=spark.spark,
        parsed_path=source['uniprot_evidence'],
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    )

    logger.info(f'write uniprot literature evidence to {destination}')
    result_df.write.parquet(destination, mode='overwrite')
```

- [ ] **Step 10.4: Run the tests**

Run: `uv run pytest test/test_uniprot_literature.py -v`
Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add src/pts/pyspark/uniprot_literature.py test/test_uniprot_literature.py
git commit -m "feat(uniprot-evidence): uniprot_literature pyspark task"
```

---

## Task 11: Register `uniprot_literature` and retarget its postprocess

**Files:**
- Modify: `config.yaml`

- [ ] **Step 11.1: Add the `uniprot_literature:` top-level step**

Insert after the `uniprot_evidence_parse:` block from Task 7:

```yaml
  #: UNIPROT LITERATURE STEP :######################################################################
  uniprot_literature:
    - name: pyspark uniprot_literature
      pyspark: uniprot_literature
      source:
        uniprot_evidence: intermediate/evidence/uniprot/uniprot_evidence.parquet
        ontoma_disease_label_lut: intermediate/ontoma/disease_label_lookup_table.parquet
        ontoma_disease_id_lut: intermediate/ontoma/disease_id_lookup_table.parquet
      destination: intermediate/evidence/uniprot_literature.parquet
  ##################################################################################################
```

- [ ] **Step 11.2: Retarget `evidence_postprocess_uniprot_literature`**

In `config.yaml`, locate the existing `evidence_postprocess_uniprot_literature:` block (around line 1015). Replace these two lines:

Before:
```yaml
        evidence_path: input/evidence/uniprot_literature.json.gz
```
```yaml
        evidence_format: json
```

After:
```yaml
        evidence_path: intermediate/evidence/uniprot_literature.parquet
```
```yaml
        evidence_format: parquet
```

Leave all other settings (`datasource_id`, `score_expression`, `unique_fields`, `failed_evidence`, `evidence`) unchanged.

- [ ] **Step 11.3: Verify config parses**

Run: `uv run pts -h`
Expected: command exits 0 and `uniprot_literature` is listed.

- [ ] **Step 11.4: Commit**

```bash
git add config.yaml
git commit -m "feat(config): register uniprot_literature step and retarget postprocess"
```

---

## Task 12: `uniprot_variants` PySpark task

**Files:**
- Create: `src/pts/pyspark/uniprot_variants.py`
- Create: `test/test_uniprot_variants.py`

- [ ] **Step 12.1: Write the failing test**

Create `test/test_uniprot_variants.py`:

```python
"""Tests for the uniprot_variants pyspark task."""

from __future__ import annotations

from pyspark.sql import Row


def _parsed_row(
    accession='P38398',
    diseases=None,
    variants=None,
):
    return Row(
        id='BRCA1_HUMAN',
        accession=accession,
        geneNames=['BRCA1'],
        diseases=diseases or [],
        variants=variants or [],
    )


def _variant(
    ft_id='VAR_007800',
    description='in BROVCA1; dbSNP:rs28897696',
    aa='p.Arg1699Gln',
    rsid='rs28897696',
    linked_omim=('604370',),
    pmids=('9145676',),
):
    return Row(
        ftId=ft_id,
        description=description,
        aminoacidChange=aa,
        dbSnpRsId=rsid,
        linkedOmimIds=list(linked_omim),
        evidencePmids=list(pmids),
    )


def _disease(omim='604370', name='Breast-ovarian cancer 1', acronym='BROVCA1'):
    return Row(
        omimId=omim,
        name=name,
        acronym=acronym,
        description='Cancer.',
        evidencePmids=['7545954'],
    )


def test_uniprot_variants_projection_and_origin(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_variants as mod

    parsed = [
        _parsed_row(
            diseases=[_disease()],
            variants=[
                _variant(),  # germline (rsid not in census)
                _variant(ft_id='VAR_999', rsid='rs99999', linked_omim=('604370',)),
            ],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed).write.parquet(str(parsed_path))

    census_path = tmp_path / 'census.txt'
    census_path.write_text('rs99999\n')

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        somatic_census_path=str(census_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    rows = df.collect()
    by_rsid = {r['variantRsId']: r for r in rows}

    assert by_rsid['rs28897696']['datatypeId'] == 'genetic_association'
    assert by_rsid['rs28897696']['alleleOrigins'] == ['germline']
    assert by_rsid['rs28897696']['datasourceId'] == 'uniprot_variants'
    assert by_rsid['rs28897696']['variantAminoacidDescriptions'] == ['p.Arg1699Gln']
    assert by_rsid['rs28897696']['diseaseFromSourceId'] == 'OMIM:604370'
    assert by_rsid['rs28897696']['literature'] == ['9145676']

    assert by_rsid['rs99999']['datatypeId'] == 'somatic_mutations'
    assert by_rsid['rs99999']['alleleOrigins'] == ['somatic']


def test_uniprot_variants_drops_unlinked(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_variants as mod

    parsed = [
        _parsed_row(
            diseases=[_disease()],
            variants=[
                _variant(ft_id='VAR_lonely', linked_omim=(), rsid='rs1'),
            ],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed).write.parquet(str(parsed_path))

    census_path = tmp_path / 'census.txt'
    census_path.write_text('')

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        somatic_census_path=str(census_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    assert df.count() == 0
```

- [ ] **Step 12.2: Run the tests to verify they fail**

Run: `uv run pytest test/test_uniprot_variants.py -v`
Expected: `ModuleNotFoundError: No module named 'pts.pyspark.uniprot_variants'`.

- [ ] **Step 12.3: Implement the task**

Create `src/pts/pyspark/uniprot_variants.py`:

```python
"""Evidence parser for UniProt disease-association variant features."""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session
from pts.pyspark.evidence_utils.uniprot import (
    DATASOURCE_VARIANTS,
    DATATYPE_GENETIC_ASSOCIATION,
    DATATYPE_SOMATIC_MUTATIONS,
    confidence_from_literature,
    load_somatic_rsids,
    uniprot_urls_struct_array,
)


def _compute_variants(
    spark: SparkSession,
    parsed_path: str,
    somatic_census_path: str,
    disease_label_lut_path: str,
    disease_id_lut_path: str,
) -> DataFrame:
    parsed = spark.read.parquet(parsed_path)
    exploded_variants = parsed.select(
        f.col('accession'),
        f.col('diseases'),
        f.explode('variants').alias('variant'),
    )

    with_linked = exploded_variants.filter(f.size(f.col('variant.linkedOmimIds')) > 0)
    exploded_omim = with_linked.select(
        f.col('accession'),
        f.col('diseases'),
        f.col('variant'),
        f.explode(f.col('variant.linkedOmimIds')).alias('omimId'),
    )

    # Resolve disease name from the entry's diseases array by matching omimId
    resolved = exploded_omim.withColumn(
        'disease',
        f.element_at(
            f.filter(f.col('diseases'), lambda d: d['omimId'] == f.col('omimId')),
            1,
        ),
    ).filter(f.col('disease').isNotNull())

    somatic = load_somatic_rsids(spark, somatic_census_path).withColumn(
        'isSomatic', f.lit(True)
    )
    joined = resolved.join(
        somatic,
        resolved['variant.dbSnpRsId'] == somatic['dbSnpRsId'],
        'left',
    ).drop(somatic['dbSnpRsId'])

    projected = joined.select(
        f.lit(DATASOURCE_VARIANTS).alias('datasourceId'),
        f.when(f.col('isSomatic').isNotNull(), f.lit(DATATYPE_SOMATIC_MUTATIONS))
            .otherwise(f.lit(DATATYPE_GENETIC_ASSOCIATION))
            .alias('datatypeId'),
        f.col('accession').alias('targetFromSourceId'),
        f.col('disease.name').alias('diseaseFromSource'),
        f.concat(f.lit('OMIM:'), f.col('omimId')).alias('diseaseFromSourceId'),
        f.col('variant.dbSnpRsId').alias('variantRsId'),
        f.array(f.col('variant.aminoacidChange')).alias('variantAminoacidDescriptions'),
        f.col('variant.evidencePmids').alias('literature'),
        confidence_from_literature(f.col('variant.evidencePmids')).alias('confidence'),
        uniprot_urls_struct_array(f.col('accession')).alias('urls'),
        f.when(f.col('isSomatic').isNotNull(), f.array(f.lit('somatic')))
            .otherwise(f.array(f.lit('germline')))
            .alias('alleleOrigins'),
    )

    logger.info('map uniprot variant diseases to EFO')
    mapped = add_efo_mapping(
        spark=spark,
        evidence_df=projected,
        disease_label_lut_path=disease_label_lut_path,
        disease_id_lut_path=disease_id_lut_path,
    )
    return mapped


def uniprot_variants(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='uniprot_variants', properties=properties)
    logger.info(f'load data from {source}')

    result_df = _compute_variants(
        spark=spark.spark,
        parsed_path=source['uniprot_evidence'],
        somatic_census_path=source['somatic_census'],
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    )

    logger.info(f'write uniprot variant evidence to {destination}')
    result_df.write.parquet(destination, mode='overwrite')
```

- [ ] **Step 12.4: Run the tests**

Run: `uv run pytest test/test_uniprot_variants.py -v`
Expected: both tests PASS.

- [ ] **Step 12.5: Commit**

```bash
git add src/pts/pyspark/uniprot_variants.py test/test_uniprot_variants.py
git commit -m "feat(uniprot-evidence): uniprot_variants pyspark task with somatic flag"
```

---

## Task 13: Register `uniprot_variants` and retarget its postprocess

**Files:**
- Modify: `config.yaml`

- [ ] **Step 13.1: Add the `uniprot_variants:` top-level step**

Insert after the `uniprot_literature:` block from Task 11:

```yaml
  #: UNIPROT VARIANTS STEP :########################################################################
  uniprot_variants:
    - name: pyspark uniprot_variants
      pyspark: uniprot_variants
      source:
        uniprot_evidence: intermediate/evidence/uniprot/uniprot_evidence.parquet
        somatic_census: input/evidence/uniprot/uniprot_somatic_census.txt
        ontoma_disease_label_lut: intermediate/ontoma/disease_label_lookup_table.parquet
        ontoma_disease_id_lut: intermediate/ontoma/disease_id_lookup_table.parquet
      destination: intermediate/evidence/uniprot_variants.parquet
  ##################################################################################################
```

- [ ] **Step 13.2: Retarget `evidence_postprocess_uniprot_variants`**

In `config.yaml`, locate the existing `evidence_postprocess_uniprot_variants:` block (around line 983). Replace these two lines:

Before:
```yaml
        evidence_path: input/evidence/uniprot_variants.json.gz
```
```yaml
        evidence_format: json
```

After:
```yaml
        evidence_path: intermediate/evidence/uniprot_variants.parquet
```
```yaml
        evidence_format: parquet
```

- [ ] **Step 13.3: Verify config parses**

Run: `uv run pts -h`
Expected: exits 0 and `uniprot_variants` is listed.

- [ ] **Step 13.4: Commit**

```bash
git add config.yaml
git commit -m "feat(config): register uniprot_variants step and retarget postprocess"
```

---

## Task 14: End-to-end local run + GCS sample reconciliation

This task is verification-heavy and requires the user's GCS sample dump in hand. Treat each step as a checkpoint: if a discrepancy is found, fix it before moving on.

**Files:**
- Modify (potentially): `src/pts/pyspark/uniprot_variants.py`, `src/pts/pyspark/uniprot_literature.py`, `src/pts/pyspark/evidence_utils/uniprot.py` — only if reconciliation against the sample reveals schema or value mismatches.

- [ ] **Step 14.1: Run the full test suite**

Run: `uv run pytest test -v`
Expected: all tests pass, including the existing suite (no regressions).

- [ ] **Step 14.2: Lint check**

Run: `uv run ruff check src/pts/transformers/uniprot_evidence.py src/pts/pyspark/uniprot_variants.py src/pts/pyspark/uniprot_literature.py src/pts/pyspark/evidence_utils/uniprot.py`
Expected: no findings.

- [ ] **Step 14.3: Run the parser locally**

Ensure `input/target/uniprot/uniprot.txt.gz` is present locally (from a prior `target_inputs` run or manual placement). Then:

Run: `uv run pts --step uniprot_evidence_parse`
Expected: completes in ~5-10 minutes; produces `intermediate/evidence/uniprot/uniprot_evidence.parquet`.

Inspect a sample:

```bash
uv run python -c "
import polars as pl
df = pl.read_parquet('intermediate/evidence/uniprot/uniprot_evidence.parquet')
print('rows:', df.height)
print('with disease:', df.filter(pl.col('diseases').list.len() > 0).height)
print('with variant:', df.filter(pl.col('variants').list.len() > 0).height)
print(df.filter(pl.col('id') == 'BRCA1_HUMAN').row(0, named=True))
"
```

Expected: a reasonable row count (≈500k entries total, ~5k with diseases, ~30k with variants); the BRCA1 row shows the BROVCA1 disease and known variants.

- [ ] **Step 14.4: Run the literature step locally**

Ensure `intermediate/ontoma/disease_label_lookup_table.parquet` and `..._id_lookup_table.parquet` exist (run the `ontoma_lut_generation` step beforehand if not).

Run: `uv run pts --step uniprot_literature`
Expected: completes; produces `intermediate/evidence/uniprot_literature.parquet`.

Inspect:

```bash
uv run python -c "
from pyspark.sql import SparkSession
s = SparkSession.builder.master('local[1]').getOrCreate()
df = s.read.parquet('intermediate/evidence/uniprot_literature.parquet')
df.printSchema()
print('rows:', df.count())
df.show(5, truncate=False)
"
```

- [ ] **Step 14.5: Run the variants step locally**

Run: `uv run pts --step uniprot_variants`
Expected: completes; produces `intermediate/evidence/uniprot_variants.parquet`.

Inspect with the same pattern as 14.4. Confirm the `alleleOrigins` and `datatypeId` distributions are non-trivial.

- [ ] **Step 14.6: Reconcile against the GCS sample dump**

The user will provide the schema (and ideally a small content sample) from:
- `gs://open-targets-pipeline-runs/ds/26.03-test5/input/evidence/uniprot_variants.json.gz`
- `gs://open-targets-pipeline-runs/ds/26.03-test5/input/evidence/uniprot_literature.json.gz`

For each output:

1. **Schema diff.** Compare `df.printSchema()` from step 14.4/14.5 with the GCS sample's schema. Note any field present in the GCS sample but missing from PTS output, or vice versa.
2. **Field-value spot check.** Pick three accessions present in both (suggested: `P38398` BRCA1, `P04637` TP53, `P51587` BRCA2). For each, compare:
   - row count
   - `datatypeId` values
   - `alleleOrigins` for variants
   - `confidence` distribution
3. **Apply targeted fixes** to `uniprot_variants.py` / `uniprot_literature.py` / `evidence_utils/uniprot.py` if the diff shows mismatches. Most likely areas:
   - Missing fields → add to the projection in the relevant `_compute_*` function.
   - Different field naming → rename in the projection.
   - Different `datatypeId` literal for germline → update `DATATYPE_GENETIC_ASSOCIATION` constant in `evidence_utils/uniprot.py`.
4. Re-run the relevant test suite and step after each fix.

- [ ] **Step 14.7: Commit any reconciliation changes**

```bash
git add src/pts/pyspark
git commit -m "fix(uniprot-evidence): align output schema with GCS reference"
```

If no changes were needed, skip this step.

- [ ] **Step 14.8: Run the downstream postprocess steps as a smoke test**

Run: `uv run pts --step evidence_postprocess_uniprot_literature`
Run: `uv run pts --step evidence_postprocess_uniprot_variants`
Expected: both complete; produce `output/evidence_uniprot_literature` and `output/evidence_uniprot_variants`.

- [ ] **Step 14.9: Final test pass**

Run: `uv run pytest test`
Expected: all tests pass.

---

## Out of scope (for follow-up)

- An Otter `download` task for `input/target/uniprot/uniprot.txt.gz` (currently obtained by the `target_inputs` flow).
- Automatic refresh of `uniprot_somatic_census.txt`. Today it's checked-in and updated manually. A future step could rebuild it from a curated UniProt resource if one becomes available.
- Excluded-row buckets (`excluded/evidence/uniprot_*`) from the new evidence steps themselves — `evidence_postprocess` already produces these from its own QC.
