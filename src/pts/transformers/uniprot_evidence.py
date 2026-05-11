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
_VARIANT_POS_RE = re.compile(r'^VARIANT\s+(?P<pos>\d+)(?:\s*\.\.\s*\d+)?\s*$')
_AA_CHANGE_RE = re.compile(r'^(?P<from>[A-Z])\s*->\s*(?P<to>[A-Z])$')
# Splits a joined qualifier text at each `/key="` token. The lookbehind
# `(?<![A-Za-z0-9_])` prevents splitting on `/key=` patterns that appear
# inside a qualifier value (e.g. immediately after a digit in a dbSNP
# identifier inside a /note value).
_QUALIFIER_SPLIT_RE = re.compile(r'(?=(?<![A-Za-z0-9_])/[a-z_]+=")')
_QUALIFIER_KV_RE = re.compile(r'^/([a-z_]+)="(.*)"\s*$', re.DOTALL)

_AA_THREE_LETTER = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
}


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


def _format_aa_change(position: str, change_text: str) -> str:
    """Convert a single-letter `X -> Y` change at a position into HGVS-like `p.AbcNNNXyz`.

    Returns '' when either residue isn't one of the 20 standard amino acids
    or the change syntax can't be parsed.
    """
    m = _AA_CHANGE_RE.match(change_text.strip())
    if not m:
        return ''
    one_from = _AA_THREE_LETTER.get(m.group('from'), '')
    one_to = _AA_THREE_LETTER.get(m.group('to'), '')
    if not one_from or not one_to:
        return ''
    return f'p.{one_from}{position}{one_to}'


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


def _parse_variant_qualifiers(position: str, qualifier_text: str) -> dict | None:
    """Parse the joined `/note=... /id=... /db_snp=...` block of one FT VARIANT.

    `qualifier_text` is the concatenated continuation content (FT lines with
    the leading `FT   ` strip stripped and joined by a single space). Returns
    a variant dict, or None when no `/id` qualifier was found.
    """
    accumulated = qualifier_text.strip()
    # Split on the start of each qualifier (a `/key="` token preceded by either
    # the start of the string or non-identifier whitespace). UniProt qualifier
    # blocks always begin a new qualifier with this pattern.
    parts = _QUALIFIER_SPLIT_RE.split(accumulated)
    qualifiers: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part.startswith('/'):
            continue
        key_match = _QUALIFIER_KV_RE.match(part)
        if key_match:
            qualifiers[key_match.group(1)] = key_match.group(2)

    ft_id = qualifiers.get('id', '')
    if not ft_id:
        return None

    note = qualifiers.get('note', '')
    db_snp = qualifiers.get('db_snp')
    evidence_text = qualifiers.get('evidence', '')
    evidence_pmids = _ECO_PUBMED_RE.findall(evidence_text)

    # Note format is typically `<change> (<description>)`. The trailing close
    # paren may or may not be present at qualifier end.
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
        'description': description,
        'aminoacidChange': aa_change,
        'dbSnpRsId': db_snp,
        'linkedOmimIds': [],
        'evidencePmids': evidence_pmids,
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
    gn_lines: list[str] = []

    ft_in_variant = False
    ft_position = ''
    ft_qualifier_lines: list[str] = []

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

        elif code == 'FT':
            ft_content = raw[5:].rstrip()
            # New feature line: starts at column 5 (no leading spaces in content)
            if ft_content and not ft_content.startswith(' '):
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
                ft_qualifier_lines.append(ft_content.strip())

    _flush_disease()

    if ft_in_variant:
        parsed = _parse_variant_qualifiers(ft_position, ' '.join(ft_qualifier_lines))
        if parsed is not None:
            variants.append(parsed)

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

    _link_variants_to_diseases(diseases, variants)

    return {
        'id': entry_id,
        'accession': accession,
        'geneNames': gene_names,
        'diseases': diseases,
        'variants': variants,
    }
