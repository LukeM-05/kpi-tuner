# Glossary

Terms used across the τ-bench 2.0 docs, in alphabetical order.

## Benchmark manifest

A JSON file under `data/tau2/benchmarks/` that defines a released benchmark:
its version, call mode (text or voice), the frozen task subset for each domain,
and the list of [cells](#cell) with their [binding revisions](#binding-revision). Three manifests ship with 2.0:

| File | Benchmark version | Mode | Cells |
|---|---|---|---|
| `text.json` | `text-v1` | text | 6 languages × 3 domains = 18 |
| `voice.json` | `voice-v2` | voice | 6 languages × 3 domains = 18 |
| `voice-v1.json` | `voice-v1` | voice | English × 3 domains = 3 |

`tau2 benchmarks list` prints them. `tau2 run --benchmark <name>` selects one by
file name (`text`, `voice`, `voice-v1`).

## Binding revision

A SHA-256 fingerprint of everything a cell feeds into a call: the [pack](#pack) revision, the localized tasks, database and policy, the
prompt and guideline versions, and the call mode. A text cell and a voice cell
for the same language and domain have different binding revisions. A run refuses
to start if the current checkout no longer produces the binding revision the
manifest records.

## Cell

One language × domain pair in a benchmark, for example `es-retail`. A
benchmark run saves each cell in its own directory (`<root>/es-retail/`). Scores are
averaged over tasks within a cell, then over cells with equal weight.

## Frame

The full, fixed population a benchmark is scored over: every cell, every
frozen task ID, and the trial indices. A run that covers only part of the
frame (a slice or a `--num-tasks` sample) has no official score; the frame is the
denominator. A saved run records its frame in `benchmark-frame.json` at the root.

## Generation quality

The public name for the τ-i18n measure that papers, data and code call
**Experience** (`experience`). It is the share of aligned agent utterances
that pass both the naturalness and the speech-fidelity checks. The
leaderboard shows it as Generation quality. Keys such as `experience` and
the paper's `experience.json` keep the old name.

## Judge

A model-based diagnostic run after the simulation, with `tau2 judges suite`. Judges never change the task reward. Each judge is defined
by a contract (model, prompt version, rubric version, scope) in
`src/tau2/judges/production_judges.json`. For voice-v2 submissions the judges
must be run before `tau2 submit prepare`.

## Measure

One named quantity a judge reports, such as `naturalness`, `speech_fidelity`
or `tool_use`. A measure applies to some languages and not
others; `tau2 judges summarize` lists, for each cell and measure, how many calls
it applies to, how many were judged, and the pass and fail counts.

## Pack

A language pack: `data/tau2/multilingual/<lang>/pack.yaml` plus its
localization files. It holds the caller personas, prompt clauses, voice settings,
and the localized tasks and database for each domain it covers. Every run
resolves a pack; with no `--language`, the English pack is used. See
[Language packs](language-packs.md).

## Pin

A value recorded in a manifest so that later runs can be checked against it: the subset digest of each task subset and the binding revision of
each cell. `tau2 benchmarks pin <name>` recomputes pins from the checkout and
refuses to change them unless you also pass a new `--revision`.

## Trial

One run of one task. `--num-trials N` runs each task N times with different
seeds. The released benchmarks are scored at one trial (pass@1);
`pass^k` is the share of tasks passed in all k trials.

## τ-i18n / τ-multi

The multilingual voice benchmark (`voice-v2`). Older docs and identifiers call it τ-Multilingual or "multilingual".

## τ-entity

The entity-capture benchmark on the `intake` domain. Older docs and identifiers call it "elicitation". See [τ-entity](elicitation.md).
