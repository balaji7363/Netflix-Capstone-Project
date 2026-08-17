# Netflix Movies and TV Shows Clustering

An unsupervised machine learning project that explores Netflix's content catalog (as of 2019) and groups similar titles together using text-based feature engineering and K-Means clustering.

## Business Context

This dataset consists of TV shows and movies available on Netflix as of 2019, collected from Flixable. In 2018, Flixable reported that the number of TV shows on Netflix had nearly tripled since 2010, while the number of movies had decreased by more than 2,000 titles. This project explores the dataset, engineers text-based features from titles, genres, cast, and descriptions, and clusters similar content — useful for recommendation systems, content-gap analysis, and catalog strategy.

## Dataset

- **File:** `NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv`
- **Rows:** 7,787 titles
- **Columns:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in` (genres), `description`

## Project Workflow

1. **Setup & Data Loading** — Load the dataset and inspect its shape and structure.
2. **Exploratory Data Analysis (EDA)**
   - Missing value analysis
   - Content distribution by rating, type (Movie vs TV Show), and release year
   - Trend analysis: content added by year and type
   - Top countries producing content and their type/rating composition
   - Top genres on the platform
   - Hypothesis testing: correlation between country/director and rating
3. **Data Clean-up**
   - Missing value imputation (categorical fields filled with 'Unknown', rating filled with mode, rows with missing `date_added` dropped)
   - Outlier review for `duration` and `release_year` (kept as legitimate data, only parsing failures dropped)
4. **Feature Engineering**
   - Binary encoding of content type
   - Temporal features (`year_added`, `month_added`)
   - Derived numeric features: number of genres, cast size, content age
   - Combined text "tags" feature (genre + description + director + cast + country) as the core input for clustering
5. **Pre-processing**
   - TF-IDF vectorization of the combined text feature
   - Feature scaling of numeric features
   - Dimensionality reduction using Truncated SVD
6. **Model Implementation**
   - K-Means clustering (primary algorithm — fast, scalable, interpretable)
   - Cluster count validated using the Elbow method, Silhouette score, and a hierarchical clustering dendrogram (on a sample)
   - 2D visualization of clusters via PCA
7. **Model Explainability**
   - Top TF-IDF keywords per cluster centroid
   - Cluster composition summary (dominant type, average release year, top genre)
8. **Conclusion & Business Recommendations**
   - Insights on catalog composition trends
   - How content acquisition, recommendation, and marketing teams can use the cluster output

## Key Insights

- The Netflix catalog (as of 2019) is majority movies, but the share of TV shows among **newly added** titles has been rising, consistent with Flixable's 2018 finding.
- Country and rating show a meaningful but non-deterministic relationship — content tone varies by region.
- Combining genre, description, cast, director, and country into a single text feature, then vectorizing with TF-IDF and reducing dimensionality with Truncated SVD, produced clusters that are interpretable and useful for downstream applications like recommendations and catalog strategy.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn (TF-IDF, StandardScaler, TruncatedSVD, KMeans, PCA, silhouette_score), scipy (hierarchical clustering)
- **Environment:** Google Colab / Jupyter Notebook

## Repository Structure

```
├── Netflix_Clustering_Project.ipynb        # Main notebook with full analysis
├── NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv   # Dataset
└── README.md                               # Project overview (this file)
```

## How to Run

1. Clone this repository:
   ```
   git clone <your-repo-url>
   ```
2. Open `Netflix_Clustering_Project.ipynb` in Jupyter Notebook or Google Colab.
3. Ensure the CSV file path in the notebook points to the dataset location (update the path if not using Google Drive).
4. Run all cells sequentially.

## Author

*Balaji C*
