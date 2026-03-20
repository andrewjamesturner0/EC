# Methods: Empirical Cycling Podcast Thematic Analysis

## Overview

This document describes the methodology used to transform approximately 180 podcast episodes (~3.1 million words of spoken content) into a structured, 44,000-word training knowledge base. The pipeline has four phases: transcription, clustering, multi-pass synthesis, and verification. Each phase addresses a specific challenge posed by the scale and structure of the corpus.

The fundamental problem is context: no existing language model can process 3.1 million words at once. The corpus must be reduced, but naive reduction (e.g., random sampling, chronological chunking) loses the thematic connections that make synthesis valuable. The approach taken here uses unsupervised topic clustering to create semantically coherent subsets that fit within a model's context window, then applies a three-pass extraction-synthesis-editorial pipeline that progressively compresses the corpus while preserving specificity and source traceability.

---

## Phase 0: Transcription

### Approach

All episodes were transcribed using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with the large-v3 model, run on an NVIDIA H200 GPU with 8 parallel workers. Each worker processes one episode independently, achieving approximately 28x real-time throughput across the full corpus.

### Corruption Detection and Remediation

Whisper large-v3 produces hallucination loops — segments where a single phrase repeats hundreds or thousands of times — in a significant fraction of long-form audio. In this corpus, 78 of 191 transcripts (41%) showed detectable corruption, with 40 classified as severe (>50% repeated lines).

The root cause is Whisper's autoregressive segment conditioning: each segment is decoded using the previous segment's output as context. When the model hallucinates a repeated line, that repetition becomes the context for the next segment, creating a self-reinforcing feedback loop. Once entered, the loop is deterministic at low temperatures and nearly impossible to escape.

The solution is to disable previous-text conditioning entirely using the `-mc 0` (max-context 0) flag. This forces each segment to be decoded independently, breaking the feedback loop at the cost of slightly less coherent cross-segment transitions — a negligible trade-off for podcast transcription where segment-level independence is natural.

Corruption detection uses a simple consecutive-repeated-line counter: any transcript with 3 or more consecutive identical lines is flagged. This is run as an automated post-processing step after transcription. After re-transcription with `-mc 0`, all 191 transcripts passed clean.

### Boilerplate Removal

The first 15 lines of each transcript are stripped before analysis. These lines typically contain podcast introductions, sponsor reads, and formulaic greetings that are consistent across episodes and would otherwise dominate term-frequency statistics in the clustering phase.

---

## Phase 1: Topic Clustering

### Purpose

The corpus is too large to process as a whole, but arbitrary chunking (by episode number, by series, by date) would separate thematically related episodes. For instance, the podcast returns to VO2max training across episodes spanning three years and two different series — a chronological split would lose the longitudinal thread.

Topic clustering solves this by grouping episodes with similar content, regardless of when they aired or which series they belong to. The goal is to produce clusters small enough to fit in a single model context window (~40K words per batch) while being semantically coherent enough to enable meaningful synthesis.

### Pipeline: TF-IDF → UMAP → HDBSCAN

The clustering pipeline has three stages, each addressing a different aspect of the problem.

#### Stage 1: TF-IDF Vectorisation

Each transcript is converted to a sparse vector using Term Frequency–Inverse Document Frequency (TF-IDF). TF-IDF measures how characteristic each word or phrase is of a given document relative to the full corpus. Words that appear in every episode (e.g., "training", "watts", "cycling") receive low weights; words that appear in only a few episodes (e.g., "mitochondrial biogenesis", "Copenhagen plank", "race walker") receive high weights.

**Parameters:**
- `max_features=5000`: Vocabulary limited to the 5,000 most informative terms
- `ngram_range=(1,3)`: Captures single words, bigrams, and trigrams (e.g., "sweet spot", "time to exhaustion", "critical power model")
- `sublinear_tf=True`: Applies logarithmic scaling to term frequency, reducing the dominance of highly repeated terms
- `max_df=0.85`: Excludes terms appearing in >85% of documents (corpus-wide filler)
- `min_df=2`: Excludes terms appearing in only 1 document (noise)
- `stop_words="english"`: Removes common English function words

The output is a 191 × 5,000 sparse matrix where each row is a transcript and each column is a weighted term score.

#### Stage 2: UMAP Dimensionality Reduction

The TF-IDF matrix is high-dimensional and sparse. Direct clustering on 5,000-dimensional sparse vectors is unreliable because distance metrics become less discriminative in high dimensions (the "curse of dimensionality"). UMAP (Uniform Manifold Approximation and Projection) addresses this by projecting the data into a lower-dimensional space that preserves local neighbourhood structure.

UMAP works by constructing a weighted graph of nearest neighbours in the original high-dimensional space, then optimising a low-dimensional embedding that preserves the topology of that graph. Unlike PCA (which preserves global variance), UMAP preserves *local* relationships — episodes that are similar to each other in the original space remain close in the reduced space, even if they are far from the global centroid.

**Parameters:**
- `n_components=15`: Reduced to 15 dimensions for clustering (a separate 2D projection is used for visualisation)
- `n_neighbors=15`: Each point's neighbourhood is defined by its 15 nearest neighbours. This was the key tuning parameter: lower values (5, 10) produced too many micro-clusters; higher values (25+) merged distinct topics into overly broad groups.
- `metric='cosine'`: Cosine similarity is the standard distance metric for TF-IDF vectors, as it measures angular similarity regardless of document length
- `min_dist=0.0`: Allows points to cluster tightly, which improves HDBSCAN's ability to detect dense regions

#### Stage 3: HDBSCAN Clustering

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) identifies clusters as regions of high density separated by regions of low density. Unlike k-means, HDBSCAN does not require specifying the number of clusters in advance, can find clusters of varying sizes and shapes, and explicitly identifies noise points — episodes that don't fit cleanly into any cluster.

This is important for this corpus because the number of natural topics is unknown in advance, and some episodes genuinely span multiple topics (the "uncategorized" or noise points).

**Parameters:**
- `min_cluster_size=5`: A cluster must contain at least 5 episodes. This prevents micro-clusters of 2-3 closely related episodes while allowing small but coherent groups.
- `min_samples=2`: Controls how conservative the clustering is. At 2, the algorithm is relatively liberal about identifying dense regions.
- `cluster_selection_method='leaf'`: Selects clusters from the leaves of the condensed tree rather than the most persistent clusters. This produces finer-grained thematic groups, which was important for a corpus where many cycling-specific topics overlap (e.g., "FTP testing" vs. "critical power" vs. "threshold training" are distinct topics that share vocabulary).

**Result:** 18 clusters containing 158 episodes, plus 33 noise points classified as uncategorized. Cluster sizes ranged from 5 to 17 episodes.

### Cluster Characterisation

Each cluster is characterised by its top TF-IDF terms (the terms with the highest average TF-IDF score across cluster members). These terms were used to assign human-readable labels:

| Cluster | Size | Label | Top Discriminating Terms |
|---------|------|-------|------------------------|
| 0 | 14 | Strength Training | squat, strength, gym, deadlift, leg press |
| 1 | 11 | Research & Methodology | study, research, paper, evidence, population |
| 2 | 9 | VO2max & Oxygen Uptake | vo2max, oxygen, cardiac, stroke volume, ramp |
| 3 | 5 | Critical Power & Lactate | critical power, w prime, lactate, MLSS |
| 4 | 10 | FTP & Testing | ftp, threshold, 20 minute, ramp test |
| ... | ... | ... | ... |
| noise | 33 | Uncategorized | (varied — these episodes span multiple topics) |

Two visualisations support cluster review: a 2D UMAP scatter plot (colour-coded by cluster) showing spatial separation, and a cosine similarity heatmap (sorted by cluster) showing within-cluster cohesion and between-cluster separation.

### Comparison to Dictionary-Based Quantitative Content Analysis

The clustering approach used here differs fundamentally from dictionary-based quantitative content analysis (QCA), a well-established method in communication research and the social sciences (e.g., LIWC, Yoshikoder, DICTION). Understanding the distinction clarifies why TF-IDF clustering was chosen.

**Dictionary-based QCA** counts occurrences of words belonging to predefined semantic categories — for instance, a "positive emotion" dictionary containing words like "happy", "excellent", "love", or a domain-specific dictionary of training terms like "interval", "threshold", "recovery". The researcher constructs or selects dictionaries before analysis, and the output is a score per document per category (e.g., "Episode 47 contains 3.2% threshold-related terms and 1.8% nutrition-related terms"). This approach is transparent, reproducible, and well-suited to hypothesis-driven research where the categories of interest are known in advance.

**TF-IDF clustering** is data-driven rather than theory-driven. It does not begin with predefined categories. Instead, it learns which terms discriminate between documents and groups documents by their emergent similarity profiles. The "categories" (clusters) are outputs of the analysis, not inputs. This makes it better suited to exploratory analysis of a corpus where the thematic structure is unknown — as in this project, where we could not predict in advance that the podcast's content would naturally separate into 18 distinct topics, or that coaching content would split along a practice/philosophy axis.

The key trade-offs:

| Dimension | Dictionary-Based QCA | TF-IDF Clustering |
|-----------|---------------------|-------------------|
| Category source | Predefined by researcher | Emergent from data |
| Transparency | High — dictionaries are inspectable | Moderate — requires interpreting TF-IDF weights and cluster top-terms |
| Reproducibility | High — same dictionary, same counts | High — deterministic pipeline with fixed random seed |
| Sensitivity to vocabulary | Limited to dictionary terms; misses synonyms and novel jargon | Captures all terms; weights by discriminative power |
| Domain adaptation | Requires building or adapting dictionaries for each domain | Adapts automatically to corpus vocabulary |
| Multi-topic documents | Assigns scores per category (documents can score on multiple) | Assigns to one cluster or noise (hard assignment) |
| Best suited for | Confirmatory analysis with known categories | Exploratory analysis with unknown thematic structure |

For this corpus, dictionary-based QCA would have required constructing cycling-specific dictionaries for each potential theme — a labour-intensive process that presupposes knowledge of the thematic structure the analysis is trying to discover. Moreover, the podcast's vocabulary is highly specialised and evolving (terms like "VLAmax", "over-unders", "W prime", "fat max" are domain-specific jargon unlikely to appear in any general-purpose dictionary), and many episodes share vocabulary across topics (e.g., "FTP" appears in episodes about testing, training philosophy, racing, and coaching). TF-IDF handles this naturally by weighting terms relative to their corpus-wide frequency, so "FTP" — which appears in most episodes — receives a low weight, while "W prime" — which appears only in critical power episodes — receives a high weight.

That said, dictionary-based QCA could complement this pipeline. A post-hoc validation step could use a domain dictionary to verify that clusters are internally coherent (e.g., the "Nutrition" cluster should score highly on nutrition-related terms) and that cross-cluster boundaries are meaningful. This was not implemented here, but the cluster top-terms table serves a similar — if less rigorous — function.

### Why Not Other Clustering Approaches?

- **LDA (Latent Dirichlet Allocation):** Produces topic distributions per document, not hard assignments. Useful for multi-topic documents, but less effective for creating the discrete batches needed for downstream context-window management.
- **Sentence-level embedding (e.g., SBERT):** Better at capturing semantic similarity than bag-of-words TF-IDF. However, for 15,000-word documents the aggregation challenge (how to represent an entire transcript as a single embedding) introduces its own errors, and TF-IDF's ngram capture proved sufficient for a domain-specific corpus with distinctive vocabulary.
- **Manual categorisation:** Possible for 191 episodes, but introduces subjective bias in boundary cases and doesn't scale. The automated approach also surfaces non-obvious groupings — the split of coaching content into two clusters (A and B) revealed a real thematic divide between practice-focused and philosophy-focused coaching episodes.

---

## Phase 2: Multi-Pass Synthesis

The synthesis pipeline has three passes, each operating at a different level of abstraction. This staged approach is analogous to a traditional qualitative research workflow — coding, categorising, and theorising — but adapted for LLM-based processing of a corpus too large for human annotation at scale.

### Comparison to Traditional Thematic Analysis

In Braun and Clarke's (2006) framework for thematic analysis, the process involves six phases: familiarisation, initial coding, theme searching, theme review, theme definition, and report production. This pipeline maps onto that framework with key differences driven by scale:

| Braun & Clarke Phase | This Pipeline | Key Difference |
|---------------------|---------------|----------------|
| 1. Familiarisation | Automated transcription + clustering | The researcher cannot read 3.1M words; TF-IDF + UMAP provides a structural overview equivalent to familiarisation |
| 2. Initial coding | Pass 1: Extraction | Codes are extracted by LLM rather than human annotator; extraction prompt enforces consistent coding categories |
| 3. Theme searching | HDBSCAN clustering | Themes emerge from content similarity rather than researcher interpretation; 18 initial themes |
| 4. Theme review | Pass 3: Restructuring (19 → 13) | Researcher reviews machine-generated themes and merges/splits based on domain knowledge |
| 5. Theme definition | Pass 2: Synthesis | Each theme is defined through declarative headlines and 200-300 word sections |
| 6. Report production | Pass 3: Editorial assembly | Executive summary, cross-references, and methodology note added |

The critical difference is ordering: in traditional thematic analysis, familiarisation precedes coding, and theme searching follows it. Here, clustering (theme searching) precedes extraction (coding), because the clustering provides the batching structure needed to fit content into LLM context windows. This means themes are defined by statistical content similarity before any interpretive coding occurs — a bottom-up approach that reduces confirmation bias but may miss themes that don't cluster on vocabulary (e.g., a "self-correction" theme that spans many topics).

### Pass 1: Extraction

**Purpose:** Reduce each episode from ~15,000 words of conversational transcript to ~500 words of structured, actionable content.

**Compression ratio:** Approximately 30:1.

**Input:** Raw transcripts, batched by cluster (≤40K words per batch).

**Output:** Per-episode extraction blocks containing five standardised fields:

1. **Core Message** (4-5 sentences): The episode's central thesis and takeaways.
2. **Key Training Recommendations**: Imperative, actionable items with specific numbers (percentages, durations, wattages, rep ranges).
3. **Caveats & Qualifications**: Population-specific advice, "it depends" factors, risk warnings.
4. **Contrarian or Non-Obvious Claims**: Positions that challenge conventional cycling wisdom.
5. **Evolving Views**: Instances where the host references changing their mind or contradicting earlier episodes.

**Extraction rules** enforce specificity and traceability:
- "Do intervals" is too vague; "Perform 4×8min at 105-110% FTP with 4min recovery" is acceptable.
- Numbers must be preserved exactly as stated.
- Speaker attribution is required when host and guest disagree.
- Hedging language ("probably", "the data suggests") must be preserved.
- Podcast logistics, ad reads, and extended anecdotes without training takeaways are skipped.

**Model choice:** Claude Sonnet. Extraction is a structured compression task that benefits from following a rigid template. Sonnet's lower cost ($3/$15 per million input/output tokens) and sufficient instruction-following ability make it the right choice for processing 191 episodes. Extraction agents run in parallel — up to 18 simultaneously, one per cluster — to minimise wall-clock time.

**Quality control:** Episode counts are verified after extraction using format-agnostic ID matching (`grep -oP '\d{9,}' | sort -u | wc -l`) rather than heading-format matching, because different agents produced slightly different heading formats.

### Pass 2: Synthesis

**Purpose:** Merge per-episode extractions within each theme into a coherent thematic summary. This is the stage where cross-episode patterns emerge: agreements that reinforce a recommendation, contradictions that reveal nuance, and chronological shifts that show the podcast's evolving position.

**Compression ratio:** Approximately 3:1 (from ~7,500 words of extractions per theme to ~2,500 words of synthesis).

**Input:** The complete extraction file for one theme (all episodes in that cluster).

**Output:** A thematic synthesis document with 5-10 declarative sub-sections, each containing:

- A **declarative headline** that makes a specific claim (not "About Threshold Training" but "Sweet Spot Volume Has Diminishing Returns Beyond Two Sessions Per Week").
- A **200-300 word body** that starts with the recommendation, provides supporting evidence and reasoning, and ends with caveats.
- **Recommendations**: Specific, actionable bullet points.
- **Sources**: Episode filenames for traceability.
- An **Evolution of Views** section tracing how the podcast's position on this theme changed over time.

**Synthesis rules** enforce editorial quality:
- **Reconcile contradictions.** If different episodes say different things, the synthesis must explain the nuance rather than arbitrarily picking one position.
- **Merge duplicates.** If five episodes all recommend riding easy on recovery days, this becomes one section citing all five, not five separate sections.
- **Preserve caveats.** "It depends" is often the real answer, and the conditions under which advice changes must be included.
- **Chronological awareness.** The numeric ID in filenames approximates chronological order, allowing the synthesis to track how views evolved (e.g., the host's progressively stronger advocacy for training volume over intensity across four years of episodes).

**Model choice:** Claude Opus. Synthesis requires editorial judgment — deciding which extractions to merge, how to reconcile conflicting advice, and how to frame the Evolution of Views narrative. These are judgment-intensive tasks where Opus's stronger reasoning capabilities justify the higher cost ($15/$75 per million tokens).

### Pass 3: Editorial Restructuring and Assembly

**Purpose:** Review the machine-generated thematic structure, merge and split themes based on domain knowledge, redistribute orphaned content, and assemble the final document.

This pass is the most human-in-the-loop stage of the pipeline. While clustering and extraction are largely automated, the restructuring decisions require domain-aware judgment about what constitutes a coherent theme for the target audience.

#### Step 1: Thematic Review

A review agent read all 19 synthesis files and identified structural problems:

- **Redundant themes:** Three molecular biology clusters (Biochemistry Foundations, Mitochondrial Physiology, Cell Biology & Signaling) covered the same AMPK/PGC-1alpha signaling pathways from different angles. These were merged into a single "Exercise Physiology & Molecular Adaptation" theme.
- **Artificial splits:** HDBSCAN split coaching content into two clusters ("Coaching & Goals A" and "Coaching & Goals B") based on vocabulary differences that didn't reflect a meaningful thematic divide. These were merged with coaching-relevant content from the uncategorized cluster.
- **Orphaned content:** The 33-episode uncategorized cluster contained 10 distinct topical sections (keto diets, FTP zone critique, easy riding, fiber type, cramp management, coaching philosophy, racing fitness, stimulus proxies, parent-athletes) that each belonged in a specific theme. These were distributed to their natural homes.

**Result:** 19 initial themes → 13 final themes.

The merges were:
- 3 molecular biology themes + fiber type → Exercise Physiology & Molecular Adaptation
- 2 coaching themes + uncategorized coaching/parenting → Coaching, Goal-Setting & Season Planning
- Endurance + Aerobic/Anaerobic + uncategorized easy rides/stimulus → Training Philosophy & Intensity Distribution
- FTP Testing + Critical Power + uncategorized FTP critique → Threshold, FTP & Performance Testing
- Recovery & Workout Design → Recovery & Periodization (content preserved, no merge)
- Psychology & Coaching → Psychology & Performance (coaching content redistributed)
- Nutrition, Strength, Racing updated with relevant uncategorized sections
- VO2max trimmed (overlapping sections moved to merged files)
- Equipment & Technique, Interval Workouts, Research & Methodology unchanged

Merge agents ran in parallel (7 simultaneously), each responsible for reading the source files, deduplicating overlapping content, and writing a coherent merged output.

#### Step 2: Document Assembly

The final document was assembled from three components:

1. **Framing content** (generated by an Opus agent): Executive summary, table of contents, 14 cross-references between themes, and methodology note.
2. **Theme content** (the 13 restructured synthesis files): Concatenated in logical order from foundational (Exercise Physiology) to applied (Racing & Team Tactics) to meta (Research & Methodology).
3. **Structural normalisation** (programmatic): Heading levels adjusted so themes are H2 and sub-sections are H3, ensuring consistent document hierarchy.

This split approach — LLM for editorial framing, shell scripts for assembly — avoids the failure mode of asking a language model to write a 40,000+ word document in a single output, which reliably stalls or truncates.

---

## Phase 3: Verification

### Grounding Verification

The synthesis pipeline introduces several categories of error, identified through verification of 60 claims (~10% of specific factual assertions):

1. **Factual conflation** (2 cases): The model merges details from different studies or episodes into a single claim. For example, performance metrics from the Havemann cycling study (40-70W deficit) were attributed to the Burke race walker study (which measured time differences, not watts) because both were discussed in the same podcast episode. In another case, a "30-40 watts of FTP gain from reducing endurance intensity" was synthesised from a figure that described gains in a different context (prior year under low stress).

2. **Fabricated structure** (2 cases): The model creates terminology or frameworks that sound authoritative but don't exist in the source material. A "TEAM framework (Treat Emotions As Messengers)" was attributed to a podcast guest who never used that acronym — the model invented a mnemonic for a concept it understood correctly. A "5-4-3-2-1 grounding exercise" was imported from general therapeutic knowledge and attributed to a specific podcast guest, who actually described a different sensory contact exercise.

3. **Numerical inversion** (1 case): The model reported a "~30% performance deficit" for PGC-1alpha/beta knockout mice, when the actual deficit was ~70% — the mice performed at 30% of control levels. The model confused "performing at X%" with "deficit of X%."

4. **Fabricated detail** (1 case): "10x more NRF1 and 2x mitochondrial density" was stated as a compensatory response, but the transcript says "3-fold citrate synthase and 10-fold cytochrome oxidase." NRF1 is never mentioned. The model substituted a plausible-sounding molecular biology term.

5. **Attribution error** (1 case): A sub-100W recovery ride prescription was attributed to a specific coach (Rory Porteous) but could not be confirmed in his episode transcript.

None of these errors are hallucinations in the traditional sense (inventing facts from nothing). All involve the model making the output *more coherent* than the source material — smoothing over the distinction between two studies discussed in sequence, packaging a diffuse concept into a catchy acronym, or importing well-known domain knowledge and attributing it to a specific source. These errors are particularly insidious because they read convincingly and survive casual review.

### Verification Protocol

60 specific, verifiable factual claims were sampled from the final output, spanning all 13 themes (~10% of estimated 600+ specific factual assertions). Claims were selected to include numerical values, study citations, direct attributions, and specific recommendations — the categories most susceptible to synthesis errors. For each claim:

1. The cited source episode transcript was located in the `transcripts/` directory.
2. The relevant section of the transcript was searched for supporting or contradicting evidence.
3. The claim was rated as SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED.

Verification was conducted in six batches: an initial 10-claim sample followed by five 10-claim batches covering themes 1-13. Each batch was verified by an independent agent reading the source transcripts.

**Results:**

| Rating | Count | Percentage |
|--------|-------|------------|
| Fully supported | 42 | 70% |
| Partially supported | 13 | 22% |
| Not supported (errors) | 5 | 8% |

All 5 errors were corrected in the final document. The 13 partially supported claims involve minor issues: imprecise attribution, approximate numbers stated as exact, or missing nuance from the original discussion. None materially misrepresent the source.

### Limitations of Verification

- **Sampling coverage:** 60 claims out of approximately 600+ specific factual assertions (approximately 10%). While substantially better than the initial 10-claim sample, this cannot guarantee the absence of further errors. Based on the observed 8% error rate, an estimated 20-30 additional errors may remain in the unverified 90%.
- **Transcript search limitations:** Whisper transcripts contain errors in technical terminology, speaker attribution, and numerical precision. A claim may be "supported" by a transcript that itself contains a transcription error.
- **Partial support ambiguity:** Some partially supported claims reflect genuine uncertainty about whether the specific formulation in the final document matches the source. For instance, a claim about "Jim Arnold" may refer to "Jem Arnold" (a transcription variant of the same person), and a claim quantifying "LT1 drops 10W after 2.5 hours" may be a reasonable synthesis of a qualitative discussion without those exact numbers.
- **Verification agent limitations:** The verification agents used keyword search to locate relevant transcript passages, which may miss semantically relevant content that uses different terminology. A claim rated "not found" may exist in the transcript under different phrasing.

---

## Pipeline Summary

```
Audio files (180 episodes, ~600 hours)
  │
  ├─ Phase 0: Transcription
  │   whisper.cpp large-v3, 8 parallel workers
  │   Corruption detection + remediation (-mc 0)
  │   ↓
  │
  Plain text transcripts (191 files, ~3.1M words)
  │
  ├─ Phase 1: Clustering
  │   TF-IDF (5000 features, 1-3 ngrams)
  │   → UMAP (15D, cosine, n_neighbors=15)
  │   → HDBSCAN (min_cluster_size=5, leaf selection)
  │   ↓
  │
  18 clusters + 33 noise (clusters.json)
  │
  ├─ Pass 1: Extraction (Sonnet, parallel)
  │   Per-episode structured extraction
  │   191 episodes → ~95K words of extractions
  │   Compression: ~30:1
  │   ↓
  │
  19 extraction files (output/extractions/)
  │
  ├─ Pass 2: Synthesis (Opus, parallel)
  │   Per-theme narrative synthesis
  │   ~95K words → ~51K words
  │   Compression: ~2:1
  │   ↓
  │
  19 synthesis files (output/synthesis/)
  │
  ├─ Pass 3: Restructuring (Opus, parallel)
  │   19 themes → 13 themes (merge/split/redistribute)
  │   ~51K words → ~43K words (deduplication)
  │   ↓
  │
  13 restructured synthesis files
  │
  ├─ Pass 3: Assembly
  │   Opus: executive summary, cross-references
  │   Shell: concatenation, heading normalisation
  │   ↓
  │
  Final document (output/final_output.md, 44K words)
  │
  └─ Verification
      60-claim grounding check (~10% coverage) against source transcripts
      5 errors found and corrected (8% error rate)
```

**Overall compression:** ~3,100,000 words → 44,000 words (approximately 70:1), with every recommendation traceable to specific source episodes.

---

## Suggested Improvements

The following improvements would increase the credibility, reliability, and methodological rigour of AI-agent-led thematic analysis. They are ordered roughly by impact-to-effort ratio, with the most actionable improvements first.

### 1. Multi-Agent Extraction with Inter-Rater Reliability

In traditional content analysis, inter-rater reliability (IRR) is a cornerstone of credibility: two or more independent coders analyse the same material, and their agreement rate (e.g., Cohen's kappa, Krippendorff's alpha) quantifies consistency. The current pipeline uses a single LLM agent per extraction, which is analogous to a single-coder study — a recognised weakness in qualitative research.

**Improvement:** Run extraction on each episode independently with two or more agents (ideally different models — e.g., Claude Sonnet and GPT-4o), then compute agreement on: (a) which recommendations were extracted, (b) which caveats were flagged, (c) how contrarian claims were characterised. Disagreements would be resolved by a third agent or human reviewer. The agreement rate would be reported as a reliability metric alongside the final output.

This directly addresses the fabricated-structure error type observed in verification: if one agent invents a "TEAM framework" but the other does not, the disagreement surfaces the fabrication. It also catches extraction omissions — where one agent misses a key recommendation that the other captures.

**Cost trade-off:** Approximately doubles extraction cost (~$10 → ~$20 at Sonnet rates). For a one-time analysis of a complete corpus, this is modest relative to the credibility gain.

### 2. Exhaustive Grounding Verification

The current pipeline verifies 60 of approximately 600+ factual claims — a ~10% sample. This is a meaningful improvement over the initial 10-claim check, and the 8% error rate (5 of 60) provides a useful baseline, but exhaustive coverage would further increase confidence.

**Improvement:** Automate grounding verification for every specific numerical claim in the final output. A verification agent would: (a) extract all claims containing numbers, study names, or direct attributions, (b) locate the cited source transcript, (c) search for supporting evidence, (d) flag mismatches. This could run as a batch process, with human review only for flagged discrepancies.

The automation is feasible because the claims are already structured with source citations, and the transcripts are searchable. The main challenge is matching natural-language claims to conversational transcript passages, which requires fuzzy matching and contextual search rather than exact string matching.

**Expected impact:** Based on the 60-claim sample (8% error rate), an exhaustive check of all ~600 claims might surface 20-30 additional corrections across the full document.

### 3. Transparent Provenance Chains

The current pipeline preserves source episode citations, but the chain from transcript → extraction → synthesis → final output is opaque. A reader cannot easily verify how a specific sentence in the final document relates to the original transcript.

**Improvement:** Implement a provenance system that links each final-document claim to: (a) the specific extraction block it was derived from, (b) the approximate line range in the source transcript. This could take the form of inline references (e.g., `[ep:807518419, lines 227-238]`) or a separate provenance index file mapping final-document paragraph numbers to transcript locations.

This would allow any reader to perform their own grounding verification without re-running the pipeline, and would make the analysis auditable in the sense expected by systematic review standards.

### 4. Cluster Stability Analysis

The current clustering is run once with fixed parameters. Different random seeds, UMAP configurations, or HDBSCAN parameters would produce somewhat different cluster assignments. The sensitivity of the final output to these upstream choices is unknown.

**Improvement:** Run the clustering pipeline N times (e.g., 50-100) with varied parameters (different random seeds, n_neighbors in [10, 15, 20], min_cluster_size in [4, 5, 6]) and measure cluster stability. Episodes that consistently cluster together across runs are robust assignments; episodes that move between clusters are boundary cases whose thematic assignment should be treated with lower confidence.

Techniques from ensemble clustering (e.g., consensus matrices, adjusted mutual information across runs) could quantify overall stability. Episodes with low assignment stability could be flagged in the final document, alerting readers that their thematic placement is a judgment call rather than a robust statistical result.

This is particularly important given that 33 episodes (17% of the corpus) were assigned to the noise cluster and manually redistributed — a process that was effective but unvalidated.

### 5. Extraction Recall Measurement

The current pipeline has no measure of extraction recall — the fraction of actionable recommendations in a transcript that the extraction agent actually captures. An agent might consistently extract 70% of recommendations, and the missing 30% might be systematically biased (e.g., toward caveats, minority opinions, or recommendations that contradict the episode's main thesis).

**Improvement:** Select a stratified sample of 10-15 episodes across themes and episode lengths. Have a human domain expert independently extract recommendations from the raw transcript (the "gold standard"). Compare the agent's extractions against this gold standard to measure recall, precision, and systematic biases.

This is the most labour-intensive improvement on the list, but it addresses a blind spot that no amount of automated verification can cover: you cannot verify the absence of something in the output without an independent measure of what should be there.

### 6. Synthesis Faithfulness Scoring

The synthesis pass (extraction → thematic summary) is the stage most prone to editorialising — the model may strengthen hedged claims, resolve ambiguity in one direction, or impose a narrative arc on views that were actually scattered. The current pipeline has no quantitative measure of synthesis faithfulness.

**Improvement:** Implement an automated faithfulness check using an LLM-as-judge approach. For each synthesis section, a separate agent reads both the synthesis and the underlying extraction blocks, then rates: (a) whether the synthesis accurately represents the extractions, (b) whether hedging language was preserved, (c) whether any claims were strengthened beyond what the extractions support, (d) whether contradictions between episodes were presented fairly or resolved in favour of one position.

This is related to the "LLM-as-judge" paradigm used in model evaluation (Zheng et al., 2023). The key design choice is whether the judge agent has access to the source transcripts (expensive, thorough) or only the extractions (cheaper, catches synthesis-level errors but not extraction-level ones). A two-tier approach — extraction-level judge for all themes, transcript-level judge for a sample — would balance cost and coverage.

### 7. Member Checking / Expert Review

In qualitative research, member checking involves returning findings to the original participants (or domain experts) for validation. For a podcast analysis, this would mean sharing the synthesis with the podcast host or recognised domain experts and soliciting feedback on accuracy, emphasis, and omissions.

**Improvement:** Share the final document (or individual theme sections) with cycling coaches, sport scientists, or the podcast community for structured review. Specific questions would include: "Does this accurately represent the podcast's position?", "Are any important caveats missing?", "Has any recommendation been overstated?"

This is the gold standard for credibility in qualitative research and cannot be replaced by any automated method. It is also the most logistically challenging improvement, as it requires external participation. A lightweight alternative would be to publish the document with a structured feedback mechanism and incorporate corrections iteratively.

### 8. Sensitivity Analysis Across Models

The entire synthesis was produced using Claude models (Sonnet for extraction, Opus for synthesis). Different model families have different biases in how they summarise, what they emphasise, and how they handle ambiguity. The current pipeline's outputs are entangled with Claude-specific tendencies.

**Improvement:** Re-run the synthesis pass (or a representative subset of themes) using a different model family (e.g., GPT-4o, Gemini 1.5 Pro) and compare the outputs. Systematic differences would reveal model-specific biases — for instance, one model might consistently emphasise physiological mechanisms while another emphasises practical recommendations. Cross-model agreement would increase confidence that findings are properties of the corpus rather than artifacts of the model.

This is analogous to methodological triangulation in qualitative research: using multiple methods (or in this case, multiple models) to approach the same data and checking for convergent findings.

### 9. Negative Case Analysis

The current pipeline is designed to find and synthesise patterns — recurring themes, consistent recommendations, majority positions. It does not systematically search for negative cases: episodes or passages that contradict the synthesised position, outlier recommendations, or topics conspicuously absent from the corpus.

**Improvement:** After synthesis, run a dedicated "negative case" agent that reads each theme section and then searches the full corpus for: (a) episodes that contradict the theme's recommendations, (b) important topics within the theme's scope that are never discussed, (c) recommendations that appear only once and may reflect a guest's idiosyncratic view rather than the podcast's sustained position.

In qualitative research, negative case analysis strengthens credibility by demonstrating that the researcher has actively sought disconfirming evidence rather than cherry-picking supportive passages. The "Evolution of Views" sections in the current pipeline partially serve this function (by documenting where the podcast contradicted its own earlier positions), but a systematic search for unresolved contradictions and gaps would go further.

### Summary of Improvements

| Improvement | Primary Benefit | Effort | Impact on Credibility |
|-------------|----------------|--------|----------------------|
| Multi-agent IRR | Catches fabrication, measures consistency | Moderate (2x extraction cost) | High |
| Exhaustive grounding | Catches factual errors at scale | Moderate (automated) | High |
| Provenance chains | Enables independent audit | Low-moderate | High |
| Cluster stability | Quantifies sensitivity to upstream choices | Low (computational) | Moderate |
| Extraction recall | Measures what the pipeline misses | High (requires human expert) | High |
| Synthesis faithfulness | Catches editorialising and hedging loss | Moderate (LLM-as-judge) | Moderate-high |
| Expert review | External validation of accuracy and emphasis | High (logistics) | Very high |
| Cross-model sensitivity | Separates corpus findings from model artifacts | Moderate (re-run subset) | Moderate |
| Negative case analysis | Demonstrates disconfirming evidence was sought | Low-moderate | Moderate |

None of these improvements were implemented in the current pipeline. Their inclusion would move the methodology from "exploratory AI-assisted analysis" toward "systematic AI-assisted analysis with validated reliability" — a meaningful distinction for any context where the output is used to inform training decisions, coaching practice, or further research.

---

## Adaptation for Qualitative Interview Corpora

The pipeline described above was designed for podcast episodes, where each episode typically covers one primary topic and episodes differ substantially from one another. This structural property is what makes whole-document TF-IDF clustering effective: each episode's term-frequency profile is distinctive because it reflects a single focused discussion.

Qualitative interview transcripts invert this structure. In a typical semi-structured interview study, every participant answers broadly the same questions, so each transcript covers the same set of topics. The transcripts are alike in their overall shape but contain multiple distinct themes within each one. This inversion has implications for how the pipeline would need to be adapted.

### The Unit-of-Analysis Problem

In the podcast pipeline, the unit of analysis is the whole episode. TF-IDF vectors computed at the document level are discriminative because each episode's vocabulary reflects its specific topic; an episode about VO2max uses different terms than one about nutrition.

For interviews, whole-document TF-IDF will produce near-uniform vectors. If 30 interviewees all discussed leadership, conflict resolution, and career development (because those were the interview questions), their document-level term profiles will be nearly identical. Clustering these vectors will either place all interviews in one cluster or produce clusters driven by superficial vocabulary differences (e.g., interviewees who happen to use more formal language) rather than thematic content.

The solution is to change the unit of analysis from the whole document to a *passage* or *segment* within a transcript. The pipeline needs a segmentation step between transcription and clustering.

### Segmentation Approaches

Several approaches can split interview transcripts into thematic passages:

- **Question-based segmentation:** If the interviews follow a semi-structured protocol, split at each interviewer question. Each segment then contains one question and its response. This produces the cleanest thematic units because the interview protocol itself defines topic boundaries.
- **Topic-boundary detection:** Algorithms like TextTiling (Hearst, 1997) identify topic shifts by measuring vocabulary change across a sliding window. When the vocabulary shifts significantly between adjacent blocks of text, the algorithm places a boundary. This works well for interviews with organic topic transitions.
- **Fixed-length sliding windows with overlap:** Split the transcript into overlapping windows of fixed length (e.g., 500 words with 250-word overlap). This is the most general approach (it requires no structural assumptions about the transcript) but may split mid-thought or mid-argument.
- **Turn-level segmentation:** Use speaker turns as the segmentation unit. Each interviewer-interviewee exchange pair becomes a segment. This preserves conversational coherence but may produce very short or very long segments depending on the interviewee's style.

The trade-offs: question-based segmentation produces the most meaningful units but requires structured interviews with identifiable questions in the transcript. Sliding windows work for any text but sacrifice thematic coherence at boundaries. For well-structured interviews, question-based segmentation is the clear first choice; for unstructured or conversational interviews, topic-boundary detection or turn-level segmentation is preferable.

### Clustering Passages Instead of Documents

Once the interviews are segmented into passages, the same TF-IDF → UMAP → HDBSCAN pipeline applies with minimal modification. The key change is that the pipeline now clusters *passages across all interviews* rather than whole documents. A passage from Interview 3 about conflict resolution clusters with a passage from Interview 17 about conflict resolution, which is what thematic analysis of interview data requires.

The scale changes. Where the podcast corpus had 191 whole-document units, a corpus of 30 interviews with ~10 segments each produces ~300 passage-level units. This is a similar order of magnitude and well within the pipeline's capacity. Larger studies (100+ interviews) would produce thousands of passages, potentially requiring adjusted HDBSCAN parameters (a larger `min_cluster_size` to avoid micro-clusters, for instance).

TF-IDF parameters may also need adjustment. Passage-level text is shorter than full-episode text (hundreds of words vs. thousands), so `max_features` could be reduced and `min_df` may need to be lowered to 1 to avoid discarding terms that appear in only one passage. The `max_df` threshold might need to increase, since interview passages will share more vocabulary than topically distinct podcast episodes.

### Implications for Downstream Passes

The three-pass extraction-synthesis-editorial structure transfers, but the prompts and expectations change at each stage.

**Extraction** shifts from "extract the main theme from this episode" to "extract the ideas, claims, and perspectives from these thematically grouped passages drawn from different interviewees." Because each cluster now contains passages from multiple participants on the same theme, the extraction prompt should emphasise capturing individual perspectives and noting points of agreement and disagreement across participants.

**Synthesis** benefits from the passage-clustering structure. In the podcast pipeline, synthesis must reconcile different episodes that may have discussed the same topic from different angles. In the interview pipeline, each cluster already contains multiple perspectives on the same theme, so the synthesis pass naturally produces cross-case comparison.

The **"Evolution of Views"** concept from the podcast pipeline (which tracks how the host's position changed over time) becomes cross-participant comparison: how different interviewees' perspectives on the same theme vary by role, experience, context, or demographic characteristics. If participant metadata is available (role, seniority, region, etc.), the synthesis prompt can be instructed to organise findings along these dimensions.

### LDA as an Alternative to Hard Clustering

The discussion of LDA earlier in this document noted that it "produces topic distributions per document, not hard assignments" and was dismissed for the podcast pipeline because hard cluster assignments were needed to create discrete processing batches. For interview passages, LDA becomes more attractive.

Interview passages genuinely span multiple topics more often than podcast episodes do. An interviewee answering a question about leadership might discuss both team dynamics and organisational culture in the same response. Hard HDBSCAN assignment forces this passage into one cluster; LDA assigns it a distribution (e.g., 60% team dynamics, 40% organisational culture) that better reflects its content.

A practical hybrid approach: use LDA for topic assignment, then batch passages by their *dominant* topic for the extraction pass. Passages with split assignments (e.g., 50/30/20 across three topics) could be included in multiple batches, ensuring the synthesis pass captures their contribution to each relevant theme. This increases extraction cost proportionally but avoids the information loss of hard assignment.

### What Stays the Same

The core pipeline architecture transfers:

- **Transcription** is identical. Whisper processes interview audio the same way it processes podcast audio.
- **The three-pass structure** (extraction → synthesis → editorial restructuring) applies without modification to the logic. Only the prompts change.
- **Verification** (grounding checks against source transcripts) works the same way, with the added advantage that interview passages are shorter and easier to verify than full podcast episodes.
- **The compression logic** (reducing a corpus too large for a single context window into a structured synthesis) is the same problem regardless of corpus type.

The key adaptation is inserting a segmentation step between transcription and clustering, and changing the unit of analysis from document to passage. Everything downstream of clustering operates on "units grouped by theme," and the pipeline does not care whether those units are whole episodes or interview segments.

---

## Tools and Models

| Component | Tool | Why |
|-----------|------|-----|
| Transcription | whisper.cpp (large-v3) | Open-source, GPU-accelerated, self-hosted (no API cost) |
| Clustering | scikit-learn + UMAP | Lightweight, well-documented, no compilation issues |
| Extraction | Claude Sonnet | Cost-efficient for structured compression at scale |
| Synthesis | Claude Opus | Stronger editorial judgment for reconciling contradictions and narrative construction |
| Restructuring | Claude Opus (merges) + Sonnet (simple tasks) | Match model capability to task complexity |
| Assembly | Bash + Python | Deterministic concatenation avoids LLM output limits |
| Verification | Claude Opus | Requires reading transcripts and comparing claims — a reasoning-intensive task |
| Orchestration | Claude Code | Agent parallelism, file management, pipeline state tracking |
