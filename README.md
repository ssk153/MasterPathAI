# MasterPathAI

**An interactive research analytics dashboard built with Python, machine learning, and Streamlit.**

I built MasterPathAI to explore how institutions publish and collaborate using Scopus records from 2018–2023. It combines data preparation, institution profiling, network analysis, and visualization, with an experimental recommender based on institution profile embeddings.

## What you can explore

- **Institution profiles:** compare yearly publication activity and groups with similar output patterns.
- **Collaboration networks:** explore an institution's direct collaborators and shared-paper counts.
- **Institution rankings:** compare collaboration strength, network centrality, and publication-based classification scores.
- **Experimental recommendations:** enter an interest to retrieve institutions using sentence embeddings and cosine similarity.
- **Geographic distribution:** view institution profiles using country-level coordinates.

## Dataset

The local dataset and generated outputs contain:

| Item | Count |
| --- | ---: |
| Unique papers | 20,216 |
| Author–affiliation–paper records before cleaning | 1,512,846 |
| Final institution profiles | 7,340 |
| Collaboration edge-table rows | 483,784 |
| Profiles with map coordinates | 3,920 |

Institution names come from source affiliation strings, which can include inconsistent names, departments, companies, and other organizations. These profiles are not a verified university directory, and the collection does not represent all global research or apply an engineering-only subject filter.

Processed dashboard data is included in the repository. The raw Scopus folder (~5.5 GB) and extracted author–affiliation CSV (~318 MB) are excluded through `.gitignore`; rebuilding requires the raw records separately.

## How it works

### 1. Data preparation

`_Data_Module.ipynb` parses the Scopus files and extracts paper IDs, titles, dates, authors, affiliations, affiliation IDs, and countries from `coredata` and `author-group`. It cleans text, parses dates, retains 2018–2023 records, and removes exact duplicates before producing:

| Output | Contents |
| --- | --- |
| `scopus_author_affiliation_2018_2023.csv` | Extracted author–affiliation records before cleaning; kept locally |
| `institution_paper_clean.csv` | Paper–institution–year–country groups with author-row counts |
| `institution_year_output.csv` | Unique paper counts and summed author counts per institution and year |
| `country_year_output.csv` | Unique paper counts per country and year |
| `institution_collaboration_edges.csv` | Institution pairs weighted by shared-paper counts |

Each pair of distinct institutions on a paper contributes a collaboration link. Repeated collaborations increase the weight. The notebook also plots publication trends, affiliation and country frequencies, and collaboration sizes.

Affiliation cleanup uses rules for placeholders, department names, hospitals, companies, and addresses. These rules can remove valid names or retain malformed entries; they do not resolve name variants into a single institution. Some exploratory plots count author–affiliation rows despite being labeled as paper counts; the exported yearly `num_papers` fields use unique paper IDs.

### 2. Institution profiling

`_AI_Module.ipynb` combines publication features with collaboration-network statistics.

**Publication clustering.** Yearly paper counts are pivoted into institution profiles, missing years are filled with zero, and features are standardized. K-means is evaluated for 2–8 clusters using silhouette scores, with the final output fixed at `BEST_K = 4`. The saved cluster sizes are 7,187, 151, 1, and 1, reflecting highly uneven publication-volume groups rather than research topics.

**Publication classification.** A balanced logistic regression model uses 2018–2022 counts to classify whether an institution's 2023 output reaches the 75th percentile. After evaluation on a stratified 80/20 institution split, the model is refitted on all institutions to export `top_prob` and `is_top_producer`. Ties at the threshold produce 2,296 positive institutions, rather than exactly 25%.

**Collaboration analysis.** NetworkX builds an undirected weighted graph and calculates:

- `weighted_degree`: the sum of shared-paper weights across an institution's links.
- `degree_centrality`: its number of distinct collaborators, normalized by the graph's possible neighbors.

These scores join the publication features and model outputs in `institution_ai_summary.csv`, with zero graph scores for isolated institutions. Repeated and reversed institution pairs are combined, so the graph can have fewer edges than the edge CSV has rows.

**Geographic context.** Each institution receives its most frequent country from the extracted records. A centroid lookup covering 15 countries supplies coordinates for the map.

### 3. Embeddings and recommendation

The pretrained `all-MiniLM-L6-v2` model encodes institution profiles into 384-dimensional vectors using this text format:

```text
<institution name> cluster <cluster number> degree <weighted degree>
```

Vectors are saved in `bert_embeddings.csv`. The dashboard encodes a user's interest with the same model, then returns up to ten institutions ranked by cosine similarity. The encoder is used without fine-tuning.

The recommender is experimental: profiles contain institution names and numeric signals, without paper titles, abstracts, specialties, or program descriptions. Matches need further investigation before supporting a research or master's program choice. The pipeline does not perform topic modeling or paper-level classification.

A separate balanced logistic regression model uses these embeddings to classify institutions whose collaboration strength reaches the 75th percentile. It uses a stratified 80/20 split and exports `confusion_matrix_bert.png` and `ai_model_metrics.csv`, replacing the earlier publication classifier's metrics.

## Dashboard

`_streamlit_app.py` uses Streamlit, PyVis, and PyDeck to display the saved CSV outputs.

| Page | Features |
| --- | --- |
| Overview | Institution and cluster totals, saved model metrics, and collaboration rankings shown as percentile ranks within the displayed subset |
| Top Institutions | Collaboration rankings with cluster, centrality, and publication-output probability |
| Collaboration Network | Interactive graph of direct collaborators and a shared-paper-weight table |
| AI Recommender | Interest input, similarity rankings, and institution details |
| Global Map | Bubbles sized by collaboration strength using square-root scaling and a radius cap |

The cluster filter applies only to **Top Institutions**. Yearly publication summaries are available as CSVs, but time-series views are not yet included.

Map points use country centers with small random offsets to separate overlapping bubbles. They do not represent campus locations, and profiles without coordinates are omitted.

## Results and limitations

The saved **sentence-transformer + logistic regression collaboration classifier** reports:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8903 |
| Precision | 0.7240 |
| Recall | 0.9167 |
| F1 | 0.8090 |
| ROC-AUC | 0.9654 |

Weighted degree appears in both the input text and the rule defining the target label. These results measure classification of an existing collaboration-based label and do not establish predictive ability for future research performance.

The dashboard's **`top_prob` comes from the publication classifier**, while **its displayed metrics come from the embedding-based collaboration classifier**. They describe different tasks and should not be compared as evidence of model improvement. The refitted `top_prob` scores are not validated future-year forecasts.

Affiliation quality, dataset coverage, and uneven clusters also affect the rankings. Some notebook commentary reflects earlier runs; the dashboard uses the exported CSVs.

## Run locally

Run from the project root. Dependencies are not version-pinned, so versions may need adjustment for your environment.

```bash
python -m pip install pandas numpy matplotlib scikit-learn networkx streamlit pyvis pydeck sentence-transformers jupyter
streamlit run _streamlit_app.py
```

The app requires these files in `data/`:

- `institution_ai_summary.csv`
- `institution_year_output.csv`
- `institution_collaboration_edges.csv`
- `ai_model_metrics.csv`
- `bert_embeddings.csv`

The recommender loads `all-MiniLM-L6-v2` on the first query and requires internet access unless the model is already cached.

## Rebuild the analysis

1. Place the raw records in `ScopusData2018-2023/` and keep `data/country_centroids.csv` available.
2. Open Jupyter from the project root and run `_Data_Module.ipynb` from top to bottom.
3. Run `_AI_Module.ipynb` from top to bottom.
4. Restart Streamlit or clear its data cache to load the rebuilt outputs.

Both notebooks overwrite output files. The data notebook also cleans an existing AI summary when present, so run the AI notebook afterward to rebuild dependent outputs. Cleanup does not rewrite every intermediate table, which can leave different institution counts across exports.

Rebuilding involves several gigabytes of data and substantial in-memory processing. The included outputs let you explore the dashboard without rerunning the pipeline.

## Repository layout

```text
MasterPathAI/
├── _Data_Module.ipynb          # Extraction, cleanup, aggregation, and EDA
├── _AI_Module.ipynb            # Clustering, classifiers, embeddings, and graph analysis
├── _streamlit_app.py           # Streamlit dashboard
├── data/                      # Processed tables, embeddings, metrics, and map lookup
├── lib/                       # Frontend assets for network rendering
├── ScopusData2018-2023/        # Local raw data; excluded from Git
└── README.md
```
