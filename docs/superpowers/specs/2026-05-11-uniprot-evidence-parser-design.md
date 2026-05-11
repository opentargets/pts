# UniProt Evidence Parser — Python/PySpark Port Design

## Summary

Replace the legacy Java `uniprot-evidence-parser` (which today produces the `uniprot_variants.json.gz` and `uniprot_literature.json.gz` files that PTS consumes externally) with three native PTS steps. The two evidence outputs become first-class PTS pipelines, parquet-formatted, mappable via the existing ontoma LUTs, and runnable locally with `uv run pts --step ...`.

## Steps Overview

| PTS Step | Module | Input | Output |
|----------|--------|-------|--------|
| `uniprot_evidence_parse` | `transformers/uniprot_evidence.py` (Polars) | `input/target/uniprot/uniprot.txt.gz` | `intermediate/evidence/uniprot/uniprot_evidence.parquet` |
| `uniprot_variants` | `pyspark/uniprot_variants.py` | parsed parquet + humsavar.txt + ontoma LUTs | `intermediate/evidence/uniprot_variants.parquet` |
| `uniprot_literature` | `pyspark/uniprot_literature.py` | parsed parquet + ontoma LUTs | `intermediate/evidence/uniprot_literature.parquet` |

### Dependency graph

```
input/target/uniprot/uniprot.txt.gz
            │
            ▼
   uniprot_evidence_parse
            │
            ├──────────────────────┬──────────────────────┐
            ▼                      ▼                      │
   intermediate/evidence/uniprot/uniprot_evidence.parquet │
            │                      │                      │
            ▼                      ▼                      │
   uniprot_variants ◀── humsavar.txt                      │
   uniprot_literature ◀──────────────────────── ontoma LUTs
            │                      │
            ▼                      ▼
   intermediate/evidence/uniprot_variants.parquet
   intermediate/evidence/uniprot_literature.parquet
            │                      │
            ▼                      ▼
   evidence_postprocess_uniprot_variants    (existing — retargeted)
   evidence_postprocess_uniprot_literature  (existing — retargeted)
```

`uniprot_variants` and `uniprot_literature` are independent and can run in parallel after `uniprot_evidence_parse`.

### What is NOT in scope

- The OpenTargets JSON-schema validation that the Java pipeline runs. PTS's `evidence_postprocess` performs the equivalent role downstream.
- Live OLS OMIM→EFO fetching. Replaced by the standard `add_efo_mapping` ontoma-LUT join.
- Any change to `evidence_postprocess` logic itself. Only its two `uniprot_*` config blocks are edited (path + format).
- Producing the Java pipeline's nested provenance JSON. The new outputs are flat evidence rows conforming to `pts/schemas/evidence.json`.

## Step Details

### 1. `uniprot_evidence_parse`

**Module:** `src/pts/transformers/uniprot_evidence.py`

**Purpose:** One streaming pass over the Swiss-Prot flat file, emitting an intermediate parquet that exposes both disease comments and variant features per entry. Both downstream PySpark tasks read this; the 700MB flat file is parsed only once.

**Implementation pattern:** Mirrors `transformers/uniprot.py` — `_parse_record(lines: list[str]) -> dict` per entry, `_parse_uniprot(file_obj)` streams entries delimited by `//`. Polars `DataFrame.write_parquet` for the output.

**Parsing rules:**

- **CC DISEASE blocks** (lines `CC -!- DISEASE: <name> (<acronym>) [MIM:NNNNNN]: ...`): walked with the existing CC state-machine pattern. Before brace stripping, extract `{ECO:0000269|PubMed:NNNNNN}` PMIDs into `evidencePmids`. Pull OMIM ID from `[MIM:NNNNNN]`, disease name from before the parenthesis, acronym from inside the parenthesis.
- **FT VARIANT features** (`FT VARIANT <pos>` plus continuation `FT   ` lines carrying `/note="..."`, `/id="VAR_NNN"`, `/evidence="ECO:..."`, `/db_snp="rsNNN"`). Reassemble continuation lines, then extract: `ftId`, `description` (the full `/note` content), `aminoacidChange` (parsed from the variant position + change in the FT line), `dbSnpRsId`, `evidencePmids` (from `/evidence` ECO PubMed refs).
- **Variant→disease linkage**: best-effort. For each variant, scan its description for any of the current entry's disease acronyms; matched acronyms produce `linkedOmimIds`. Variants with no acronym match remain in the intermediate but get filtered downstream. This is intentionally looser than the Java pipeline's object-graph linking; we accept that a small tail of variants will go unlinked.
- **Robustness:** malformed entries return whatever was parsed, with a WARN log. One bad record does not stop the run.

**Output schema:**

```python
pl.Schema({
    'id':         pl.String,                        # UniProt ID, e.g. BRCA1_HUMAN
    'accession':  pl.String,                        # primary accession
    'geneNames':  pl.List(pl.String),
    'diseases': pl.List(pl.Struct({
        'omimId':         pl.String,
        'name':           pl.String,
        'acronym':        pl.String,
        'description':    pl.String,
        'evidencePmids':  pl.List(pl.String),
    })),
    'variants': pl.List(pl.Struct({
        'ftId':             pl.String,
        'description':      pl.String,
        'aminoacidChange':  pl.String,
        'dbSnpRsId':        pl.String,
        'linkedOmimIds':    pl.List(pl.String),
        'evidencePmids':    pl.List(pl.String),
    })),
})
```

Field names here are internal to the intermediate; downstream Spark code projects to the final evidence schema and is free to rename.

**Config:**

```yaml
uniprot_evidence_parse:
  - name: transform uniprot evidence
    transformer: uniprot_evidence
    source: input/target/uniprot/uniprot.txt.gz
    destination: intermediate/evidence/uniprot/uniprot_evidence.parquet
```

**Tests** (`test/test_uniprot_evidence.py`): unit tests against `_parse_record` with inline flat-file fragments. Covers: single disease, multi-disease, disease with no MIM, variant with rsID, variant without rsID, variant→disease linking via acronym, ECO-PubMed extraction, FT continuation lines, malformed entry returns gracefully.

### 2. `uniprot_variants`

**Module:** `src/pts/pyspark/uniprot_variants.py`

**Signature:** `(source: dict[str, str], destination: str, settings: dict, properties: dict)` — matches every other evidence pyspark task.

**Pipeline:**

1. Read `uniprot_evidence` intermediate parquet.
2. Explode `variants`.
3. Drop rows with empty `linkedOmimIds`.
4. Explode `linkedOmimIds` → one row per (variant × OMIM).
5. Load `humsavar.txt` as a DataFrame keyed by dbSNP rsID. Left-join on `dbSnpRsId`.
6. Derive `alleleOrigins` from the humsavar CATEGORY column. **Open**: exact mapping confirmed against user-provided GCS sample; if humsavar does not yield somatic information we fall back to porting the static `uniprot_somatic_census.txt` from the Java repo as an input file.
7. Project to evidence schema (final field set to be confirmed against the user-provided GCS sample dump):
   - `datasourceId = "uniprot_variants"`
   - `datatypeId = "somatic_mutations"` when row is somatic, `"genetic_association"` otherwise — **to be confirmed against GCS sample**
   - `targetFromSourceId = accession`
   - `diseaseFromSource = disease.name`
   - `diseaseFromSourceId = "OMIM:" || omimId`
   - `variantRsId = dbSnpRsId`
   - `variantAminoacidDescriptions = array(aminoacidChange)`
   - `literature = evidencePmids` (variant-level)
   - `confidence = "high"` if `literature` non-empty else `"medium"`
   - `urls = array(struct("UniProt" AS niceName, "https://www.uniprot.org/uniprotkb/" || accession AS url))`
   - `alleleOrigins` as above
8. `add_efo_mapping(...)` to fill `diseaseFromSourceMappedId`.
9. `write.parquet(..., mode="overwrite")`.

**Config:**

```yaml
uniprot_variants:
  - name: pyspark uniprot_variants
    pyspark: uniprot_variants
    source:
      uniprot_evidence: intermediate/evidence/uniprot/uniprot_evidence.parquet
      humsavar:         input/evidence/uniprot/humsavar.txt
      ontoma_disease_label_lut: intermediate/ontoma/disease_label_lookup_table.parquet
      ontoma_disease_id_lut:    intermediate/ontoma/disease_id_lookup_table.parquet
    destination: intermediate/evidence/uniprot_variants.parquet
```

**Tests** (`test/test_uniprot_variants.py`): tiny inline DataFrames for parsed-uniprot rows, humsavar, and ontoma LUTs, called against the session fixture from `conftest.py`. Asserts projected columns and join behaviour.

### 3. `uniprot_literature`

**Module:** `src/pts/pyspark/uniprot_literature.py`

**Pipeline:**

1. Read `uniprot_evidence` intermediate parquet.
2. Explode `diseases`.
3. Filter: keep only rows where `disease.evidencePmids` is non-empty (matches Java behaviour — a literature-curated row needs at least one citing PMID).
4. Project to evidence schema:
   - `datasourceId = "uniprot_literature"`
   - `datatypeId = "genetic_association"` — **to be confirmed against GCS sample**
   - `targetFromSourceId = accession`
   - `diseaseFromSource = disease.name`
   - `diseaseFromSourceId = "OMIM:" || omimId`
   - `literature = disease.evidencePmids`
   - `confidence = "high"` if literature non-empty else `"medium"` (always "high" after the filter — kept for symmetry with variants)
   - `urls = array(struct("UniProt" AS niceName, "https://www.uniprot.org/uniprotkb/" || accession AS url))`
5. `add_efo_mapping(...)`.
6. `write.parquet(..., mode="overwrite")`.

**Config:**

```yaml
uniprot_literature:
  - name: pyspark uniprot_literature
    pyspark: uniprot_literature
    source:
      uniprot_evidence: intermediate/evidence/uniprot/uniprot_evidence.parquet
      ontoma_disease_label_lut: intermediate/ontoma/disease_label_lookup_table.parquet
      ontoma_disease_id_lut:    intermediate/ontoma/disease_id_lookup_table.parquet
    destination: intermediate/evidence/uniprot_literature.parquet
```

**Tests** (`test/test_uniprot_literature.py`): same pattern as `uniprot_variants`.

### 4. Shared utilities

`src/pts/pyspark/evidence_utils/uniprot.py`: the UniProt URL constructor, the confidence rule, and the literal `datasourceId`/`datatypeId` mapping. Imported by both pyspark tasks to avoid duplication.

## Inputs

### Existing (reused)

- `input/target/uniprot/uniprot.txt.gz` — Swiss-Prot flat file (already downloaded by the `target_inputs` flow).
- `intermediate/ontoma/disease_label_lookup_table.parquet`, `intermediate/ontoma/disease_id_lookup_table.parquet` — produced by `ontoma_lut_generation`.

### New download

- `input/evidence/uniprot/humsavar.txt` — fetched from `https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/variants/humsavar.txt`. Added as an Otter download alongside the most-similar existing evidence-input fetch.

### Fallback input (only if humsavar is insufficient)

- `input/evidence/uniprot/uniprot_somatic_census.txt` — ported verbatim from `src/bin/uniprot_somatic_census.txt` in the Java repo. Static; refreshed manually when UniProt updates the census. Decision made after inspecting the user-provided GCS sample dump.

## Integration with existing config

The two existing `evidence_postprocess_uniprot_*` blocks (config.yaml:983 and :1015) are edited:

| Field | Before | After |
|-------|--------|-------|
| `source.evidence_path` | `input/evidence/uniprot_variants.json.gz` / `..._literature.json.gz` | `intermediate/evidence/uniprot_variants.parquet` / `..._literature.parquet` |
| `settings.evidence_format` | `json` | `parquet` |

All other postprocess settings (`unique_fields`, `score_expression`, `datasource_id`, etc.) are untouched.

## Failure modes

| Scenario | Behaviour | Rationale |
|---|---|---|
| OMIM has no EFO match in ontoma LUTs | Keep row, `diseaseFromSourceMappedId = null` | Matches every other PTS evidence pipeline. |
| Variant has no linked OMIM | Drop row | No disease attribution possible; same as Java. |
| Variant rsID missing or not in humsavar | Default `alleleOrigins = ["germline"]` | Conservative default. |
| CC DISEASE block has no PMIDs | Literature row dropped, variant rows for the entry unaffected | Matches Java. |
| Variant description has no acronym match | Drop row | No way to attribute disease. |
| Malformed UniProt entry | `_parse_record` returns partial result, WARN logged, parse continues | One bad record should not kill the run. |

`Excluded` outputs are deferred to a follow-up if needed; `evidence_postprocess` already handles exclusion bookkeeping downstream.

## Local run

```bash
# one-time / per-release
uv run pts --step uniprot_evidence_parse

# either or both, in any order:
uv run pts --step uniprot_variants
uv run pts --step uniprot_literature
```

Prerequisites the developer must have produced locally beforehand:
- `input/target/uniprot/uniprot.txt.gz` (from `target_inputs`)
- `intermediate/ontoma/disease_label_lookup_table.parquet` and `_id_lookup_table.parquet` (from `ontoma_lut_generation`)
- `input/evidence/uniprot/humsavar.txt` (from the new download step, or placed manually)

Missing prerequisites surface as Otter "not found" errors — same DX as every other step.

## Open items requiring the GCS sample dump

These are explicit verification gates before the spec can be considered final:

1. **Final field set** for `uniprot_variants` and `uniprot_literature` parquets, matching the GCS samples column-for-column (including nullability and array element struct shapes).
2. **`datatypeId` values** for each row class — confirm `"somatic_mutations"` for somatic variants, and the literal value for germline variants and literature rows.
3. **Somatic vs germline determination** — whether humsavar yields it, or whether we need to port the static `uniprot_somatic_census.txt` from the Java repo.
4. **Whether `alleleOrigins` is populated in the existing GCS output at all** — if not, drop the field and skip humsavar entirely.

## Tests summary

| File | What |
|------|------|
| `test/test_uniprot_evidence.py` | Polars parser unit tests with inline flat-file fragments. |
| `test/test_uniprot_variants.py` | PySpark task with inline mini-DataFrames for parsed-uniprot, humsavar, ontoma LUTs. |
| `test/test_uniprot_literature.py` | PySpark task with inline mini-DataFrames for parsed-uniprot and ontoma LUTs. |

End-to-end parity check (not a unit test): once the GCS sample is in hand, spot-check a handful of known accessions (e.g. `BRCA1_HUMAN`, `TP53_HUMAN`) against the reference and confirm content parity within the expected update-cycle drift.
