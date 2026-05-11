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
    variants: list[dict] = []  # populated in a later task (FT VARIANT parsing)

    # CC DISEASE accumulator
    in_disease = False
    disease_lines: list[str] = []
    gn_lines: list[str] = []

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
            gn_lines.append(content)

        elif code == 'CC':
            stripped_content = content.lstrip()
            if stripped_content.startswith('-----'):
                _flush_disease()
            elif stripped_content.startswith('-!- DISEASE:'):
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

    gn_text = _strip_braces(' '.join(gn_lines))
    for segment in gn_text.split(';'):
        segment = segment.strip()
        if '=' in segment:
            key, _, value = segment.partition('=')
            if key.strip() == 'Name':
                for v in value.split(','):
                    v = v.strip().rstrip(';')
                    if v:
                        gene_names.append(v)

    return {
        'id': entry_id,
        'accession': accession,
        'geneNames': gene_names,
        'diseases': diseases,
        'variants': variants,
    }
