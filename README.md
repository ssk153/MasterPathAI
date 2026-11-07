# Institution Research Activity & Collaboration Dashboard

A Python data science project that turns Scopus publication records from 2018–2023 into institution profiles, collaboration networks, machine learning outputs, and an interactive Streamlit dashboard.

The main questions it explores are: how much do institutions publish each year, which institutions collaborate, and which institutions have similar publication patterns? It also includes an experimental text-similarity recommender.

The implementation analyzes publication activity and coauthorship. It does **not** perform research-topic modeling, abstract analysis, or paper-level document classification.

## Data and project structure

The supplied data files contain:

| Item | Count |
| --- | ---: |
| Unique papers in the extracted Scopus table | 20,216 |
| Author–affiliation–paper rows before cleaning | 1,512,846 |
| Institution profiles in the final AI summary | 7,340 |
| Rows in the collaboration edge CSV | 483,784 |
| Profiles with map coordinates | 3,920 |

These are counts from the supplied CSV files, not results from a fresh pipeline run. Institution names are extracted affiliation strings and are not guaranteed to represent distinct, verified universities. The code does not enforce an engineering-only subject filter or establish that this collection represents all global research.

The Git upload configuration excludes `ScopusData2018-2023/` (approximately 5.5 GB) and `data/scopus_author_affiliation_2018_2023.csv` (approximately 318 MB). These files remain local. A clone needs a separately supplied raw dataset to rebuild the pipeline; the remaining generated data files are included so the dashboard can use the existing results.

```text
Project_II/
├── _Data_Module.ipynb                 # Extraction, cleaning, aggregation, EDA
├── _AI_Module.ipynb                   # Clustering, classifiers, graph analysis
├── _streamlit_app.py                  # Five-page dashboard
├── ScopusData2018-2023/               # Raw Scopus records
├── data/
│   ├── scopus_author_affiliation_2018_2023.csv
│   ├── institution_paper_clean.csv
│   ├── institution_year_output.csv
│   ├── country_year_output.csv
│   ├── institution_collaboration_edges.csv
│   ├── institution_ai_summary.csv
│   ├── bert_embeddings.csv
│   ├── ai_model_metrics.csv
│   ├── confusion_matrix_bert.png
│   └── country_centroids.csv          # Required external geographic lookup
└── lib/                              # Frontend assets used for network rendering
```

## How the pipeline works

### 1. Extract and prepare publication data

`_Data_Module.ipynb` recursively attempts to parse files in `ScopusData2018-2023/` as JSON. It searches for `coredata` and `author-group` fields, then extracts paper IDs, titles, dates, author names, affiliation names/IDs, and countries. Abstracts are not extracted.

It trims text fields, parses dates, retains 2018–2023 records, and removes exact duplicate rows. It then creates:

| Output | Meaning |
| --- | --- |
| `scopus_author_affiliation_2018_2023.csv` | Extracted rows, before subsequent cleaning |
| `institution_paper_clean.csv` | One row per paper–institution–year–country group, with an author-row count |
| `institution_year_output.csv` | Unique paper counts and summed author counts per institution and year |
| `country_year_output.csv` | Unique paper counts per country and year |
| `institution_collaboration_edges.csv` | Institution pairs with a weight counting shared papers |

For each paper, every pair of distinct institution names becomes a collaboration link. A paper involving multiple institutions therefore contributes multiple links. The AI notebook later combines links in an undirected graph, so the CSV row count is not necessarily the number of unique graph edges.

The notebook also plots publication trends, affiliation/country frequencies, and collaboration sizes. Some exploratory charts labeled as paper counts actually count author–affiliation rows; the exported yearly `num_papers` columns use unique paper IDs.

The final cleaning cells apply name-based rules to the institution-year table, edge table, and an existing AI summary if present. They remove placeholder values and strings matching department, hospital, company, address, and other patterns. These rules are heuristic: broad substring matches can remove valid institutions, while malformed affiliations can remain. They do not provide canonical institution identity resolution.

### 2. Build institution features and AI outputs

`_AI_Module.ipynb` reads the yearly publication table and collaboration edges.

**Publication-pattern clustering.** It builds an institution-by-year paper-count matrix, fills missing years with zero, standardizes the features, and evaluates K-means for 2–8 clusters using silhouette scores. The final cluster count is explicitly fixed at **4**, rather than automatically selected from the scores. These are publication-volume patterns, not subject/topic clusters. The supplied summary has cluster sizes of 7,187, 151, 1, and 1, so the grouping is highly uneven.

**Historical-output classifier.** Logistic regression uses 2018–2022 publication counts to classify whether an institution's 2023 output reaches the dataset's 75th-percentile threshold. It uses a stratified 80/20 institution split, standard scaling, and balanced class weights. The model is then refitted on all institutions to export `top_prob` and `is_top_producer`. Because the label uses `>=` a percentile threshold, ties mean the positive class need not contain exactly 25% of institutions. In the saved data, 2,296 institutions are positive.

**Collaboration analysis.** NetworkX builds an undirected weighted graph. The notebook calculates:

- `weighted_degree`: the sum of shared-paper weights across an institution's collaboration links.
- `degree_centrality`: the number of distinct collaborators, normalized by the graph's possible neighbors.

These values are merged with publication counts, cluster assignments, and classifier outputs in `institution_ai_summary.csv`. Institutions without graph links receive zero graph scores.

**Country and map lookup.** Each institution is assigned its most frequent country in the extracted affiliation table. The notebook joins `country_centroids.csv` to attach country-level latitude/longitude. The supplied lookup covers 15 countries; profiles without matching coordinates are omitted from the map.

**Sentence-transformer classifier and embeddings.** A later section uses the pretrained `all-MiniLM-L6-v2` model through Sentence Transformers. It encodes synthetic text of this form:

```text
<institution name> cluster <cluster number> degree <weighted degree>
```

A second logistic regression classifier predicts whether the institution's weighted degree reaches the 75th percentile. It uses a stratified 80/20 split and balanced class weights. The sentence-transformer itself is not fine-tuned. This section exports 384-dimensional institution embeddings, overwrites `ai_model_metrics.csv` with its own evaluation, and saves a confusion-matrix image.

The two classifiers have **different targets**. The dashboard's `top_prob` comes from the historical-output classifier, while its displayed performance metrics come from the sentence-transformer collaboration classifier.

### 3. Explore the dashboard

`_streamlit_app.py` reads the saved CSV outputs; it does not rerun extraction, clustering, or classifier training.

| Page | What it shows |
| --- | --- |
| Overview | Institution and cluster totals, saved classifier metrics, and the top institutions by collaboration strength. Bar heights are percentile ranks within the displayed subset, not raw weights. |
| Top Institutions | Institutions ranked by weighted degree, with cluster, centrality, and historical-output probability. |
| Collaboration Network | A selected institution and its direct collaborators in a PyVis graph, plus a link-weight table. |
| AI Recommender | Up to ten institution matches for a typed interest, ranked by cosine similarity between the query embedding and saved institution embeddings. |
| Global Map | Institution bubbles near country centroids, sized using collaboration strength with square-root scaling and a radius cap. |

The sidebar cluster selection currently affects **only the Top Institutions page**. Other pages use the full datasets. The yearly publication and country tables are prepared by the notebooks, but the dashboard does not currently plot their time series.

## Interpreting the results

The supplied `ai_model_metrics.csv` reports the following for the sentence-transformer + logistic regression collaboration classifier:

| Metric | Saved value |
| --- | ---: |
| Accuracy | 0.8903 |
| Precision | 0.7240 |
| Recall | 0.9167 |
| F1 | 0.8090 |
| ROC-AUC | 0.9654 |

These metrics describe a label derived from existing collaboration scores. **The input text includes the same weighted degree used to define the target**, so the evaluation does not establish future research performance or independent predictive ability. It also cannot be used as evidence that this model improves on the historical-output classifier, because the tasks differ.

Additional interpretation limits:

- The recommender embeds institution names and numeric profile text, not research titles, abstracts, program descriptions, or subject expertise. Its topic matches are experimental and do not validate a university's suitability for a research interest or master's program.
- `top_prob` is an output from the historical publication classifier after fitting on the full dataset; it is not a validated forecast for a future year.
- Map positions are country centers with small random offsets for visual separation, not actual campus locations.
- Name-based affiliation cleaning and inconsistent names can distort counts, rankings, and collaboration links. The source collection's coverage also limits broader conclusions.
- Notebook comments and prose may describe older results. The exported metrics file reflects the final sentence-transformer classifier, and the CSVs are the dashboard's actual inputs.

## Running the dashboard

Run commands from the project root so relative data paths resolve correctly. The folder has no dependency lockfile or pinned environment specification. The following are the packages imported by the app and notebooks; installation and compatibility have not been freshly verified:

```bash
python -m pip install pandas numpy matplotlib scikit-learn networkx streamlit pyvis pydeck sentence-transformers jupyter
streamlit run _streamlit_app.py
```

The dashboard requires these existing files in `data/`:

- `institution_ai_summary.csv`
- `institution_year_output.csv`
- `institution_collaboration_edges.csv`
- `ai_model_metrics.csv`
- `bert_embeddings.csv`

The recommender loads `all-MiniLM-L6-v2` when a query is entered. Internet access is needed if the pretrained model is not already cached.

## Rebuilding the outputs

1. Keep the raw records in `ScopusData2018-2023/` and the external `data/country_centroids.csv` lookup available.
2. Open Jupyter from the project root and run `_Data_Module.ipynb` from top to bottom.
3. Run `_AI_Module.ipynb` from top to bottom to create the institution summary, map fields, embeddings, metrics, and confusion matrix.
4. Start or restart the Streamlit dashboard to load the rebuilt files.

Both notebooks overwrite output files. The data notebook's final cleaning cells also modify an existing AI summary when present; rerun the AI notebook afterward to rebuild dependent outputs. Those cleaning cells do not rewrite every earlier export, so raw, intermediate, and final institution counts can differ.

The raw folder is approximately 5.5 GB and the supplied output folder approximately 374 MB. Rebuilding involves substantial in-memory extraction, institution-pair generation, clustering, and embedding work. The included outputs allow dashboard exploration without rerunning those stages.
