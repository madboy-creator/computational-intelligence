for lab 2 with visual 

# Lab Assignment: Anomaly Detection and Cluster Analysis

**Course:** Computing Intelligence  
**Name:** Mugisha Leonce  
**Registration Number:** 224007172

## 📌 Lab Overview

This lab applies unsupervised machine learning techniques to the Country Schools dataset. The dataset records, for each county, how far the installation of Learner Digital Devices (LDDs) and Teacher Digital Devices (TDDs) has progressed in schools.

The objectives of the lab are:

1. **Data Preprocessing:** clean and standardize the data for modeling.
2. **Anomaly Detection:** identify counties with unusual school and installation patterns.
3. **Cluster Analysis:** group counties by school volume and installation progress.

## 📊 Dataset

- **File:** `Country Schools.csv` (50 rows × 8 columns raw, 47 counties after cleaning)
- **Columns:**
  - `SNO`: county name (renamed to `County`)
  - `TOTALSCHOOLS`: total schools in the county
  - `TotalSchoolsInstalled`: schools where installation is complete
  - `SchoolsPending`: schools awaiting installation
  - `PercentageInstalled`: installation completion rate
  - `InstalledLDDs`: learner devices installed
  - `InstalledTDDs`: teacher devices installed
  - `FID`: row identifier

## 🛠️ Methodology & Implementation

### 1. Data Preprocessing

- **Empty and summary rows:** two completely empty rows and the national `TOTAL` row (21,729 schools) were removed. Keeping `TOTAL` would have made it the largest outlier and distorted every distance calculation.
- **Type conversion:** `PercentageInstalled` was converted from text (e.g. "10.34%") to a float.
- **Cleaning:** `SNO` was renamed to `County` and names were stripped and standardized. No duplicate rows were found.
- **Redundancy check:** `InstalledTDDs` is exactly 2 × `TotalSchoolsInstalled`, and `SchoolsPending` equals `TOTALSCHOOLS − TotalSchoolsInstalled`, in every row. Both columns carry no new information and were left out of the analysis.
- **Feature selection:** `TOTALSCHOOLS`, `TotalSchoolsInstalled`, `PercentageInstalled`, `InstalledLDDs`.
- **Standardization:** `StandardScaler` (mean 0, std 1) so that `InstalledLDDs`, which is in the tens of thousands, does not dominate distance-based methods.

### 2. Anomaly Detection

Five methods were run and compared: boxplots with the IQR rule, Z-scores (|z| > 3), Isolation Forest (`contamination=0.1`), Local Outlier Factor (`n_neighbors=5`, `contamination=0.1`) and DBSCAN noise points (`eps=1.0`, `min_samples=3`).

| County | Schools | Installed % | IQR/Z | Isolation Forest | LOF | DBSCAN | Methods |
|---|---|---|---|---|---|---|---|
| Kitui | 1,238 | 96.93 | ✔ | ✔ | ✔ | ✔ | 4 |
| Makueni | 887 | 93.91 | ✔ | ✔ | ✔ | ✔ | 4 |
| Kakamega | 880 | 65.00 | | ✔ | ✔ | ✔ | 3 |
| Bungoma | 741 | 61.54 | | | ✔ | ✔ | 2 |
| Turkana | 358 | 10.34 | | ✔ | | | 1 |
| Homa Bay | 875 | 63.09 | | | ✔ | | 1 |
| Marsabit | 138 | 5.80 | | ✔ | | | 1 |

**Findings:**

- **Kitui and Makueni** were flagged by all four methods, making them the most reliable anomalies. Both are large counties with near-complete installation. Kakamega (44,338 LDDs) was flagged by three methods.
- **Turkana and Marsabit** were flagged only by Isolation Forest. They have low installation rates (10.3% and 5.8%), but no single value is extreme, so the one-column methods (IQR, Z-score) missed them.
- The flagged counties are genuine, unusual counties rather than data errors, because the consistency checks passed, so they were kept.
- Methods disagree because each defines "unusual" differently. Counties flagged by several methods deserve the most confidence.

### 3. Cluster Analysis

**Choosing k:** the elbow curve is smooth with no clear bend, and silhouette scores are close for several values (k = 2: 0.392, k = 3: 0.377, k = 8: 0.419). k = 3 was chosen because it gives groups that are easy to interpret and actionable, not because it is the best score.

**K-Means (k = 3, silhouette 0.377):**

| Cluster | Counties | Avg. total schools | Avg. installed % | Avg. LDDs | Description |
|---|---|---|---|---|---|
| 0 | 23 | 340 | 71.5 | 10,543 | Mid-sized counties, mostly well covered (e.g. Nairobi, Embu, Tana River) |
| 1 | 17 | 724 | 73.1 | 27,443 | Large counties (e.g. Kitui, Homa Bay, Kakamega) |
| 2 | 7 | 229 | 16.6 | 2,086 | Low-coverage counties: Turkana, Garissa, Mandera, Wajir, Marsabit, Samburu, Nyeri |

**Cross-checks:**

- **Hierarchical (Ward) clustering** agrees with K-Means for 41 of 47 counties. The six northern low-coverage counties (Turkana, Garissa, Mandera, Wajir, Marsabit, Samburu) form their own group. Nyeri is the one county K-Means placed with them that Ward puts in the main group, so it is the weakest member of cluster 2. Ward also moved 5 large counties into the mid-sized group.
- **DBSCAN** found one dense group of 43 counties plus four noise points (Kitui, Makueni, Kakamega, Bungoma), matching the anomaly results. It did not separate the low-coverage counties, which shows that group is defined by a modest density difference, not a hard gap.

**Interpretation:** the low-coverage group should be prioritized for the remaining installations, with a note that ranking by percentage alone ignores absolute backlog (for example Homa Bay has 323 schools pending and Turkana 321).

## 📈 Visualizations

**Correlation matrix.** `TotalSchoolsInstalled` and `InstalledTDDs` correlate at 1.00 (the redundancy found in preprocessing). The count features correlate at 0.80 to 0.89.

<img width="698" height="581" alt="01_correlation_heatmap" src="https://github.com/user-attachments/assets/9e6bb125-434b-4ca0-afc7-120030049931" />


**Boxplots (anomaly detection).** Kitui is above the whiskers on `TOTALSCHOOLS`, and Kitui and Makueni are above them on `TotalSchoolsInstalled`. `PercentageInstalled` and `InstalledLDDs` show no outliers, which is why Turkana and Marsabit (low percentage) are invisible to one-column rules.

<img width="1589" height="389" alt="02_boxplots" src="https://github.com/user-attachments/assets/2cfdffbd-1902-4572-b3fa-0b6b8b947efc" />


**Isolation Forest anomalies in PCA space.**

<img width="702" height="547" alt="03_isolation_forest_pca" src="https://github.com/user-attachments/assets/e231e647-9eaa-4a72-8455-9ce61e2c6375" />


**Choosing k.** The elbow curve is smooth with no clear bend, and silhouette is lowest at k = 5 and highest at k = 8, so k = 3 was chosen for interpretability.

<img width="986" height="393" alt="04_elbow_silhouette" src="https://github.com/user-attachments/assets/48d4d3aa-a97f-4c09-87ed-cdf022de72cf" />


**K-Means clusters (k = 3) in PCA space.** The low-coverage cluster (cyan) sits at the lower left, and Nyeri is the member closest to the main group.

<img width="683" height="528" alt="05_kmeans_pca" src="https://github.com/user-attachments/assets/7056d1dc-b493-4da0-a9d1-91f27b29b394" />


**Dendrogram (Ward linkage).** A group of 12 large counties (Kitui, Kakamega, Makueni and others) splits off first, at height about 12.7. The six northern low-coverage counties (Garissa to Wajir) form their own branch, and Nyeri joins the main group.

<img width="1389" height="490" alt="06_dendrogram" src="https://github.com/user-attachments/assets/17fe27e9-3cc1-4fc9-b8d2-ea981014b387" />


## ⚠️ Limitations

- Only 47 counties, so results are sensitive to individual counties and small clusters can change if one value changes.
- No ground-truth labels, so cluster quality is judged only by separation (silhouette about 0.38, which is moderate).
- Results depend on chosen parameters: `k`, `contamination` (which fixes roughly how many points Isolation Forest and LOF flag), `n_neighbors`, `eps` and `min_samples`.
- The count features are strongly correlated (0.80 to 0.89), and Kitui is extreme enough to skew the standardization. A log transform or `RobustScaler` would test how much this affects the results.
- The data has no population, poverty, school size or infrastructure information that could explain why some counties lag.

## 🚀 How to Run the Code

1. Install Python with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` and `scipy`.
2. Place `Country Schools.csv` next to the notebook and change the file path in the first cell (it currently points to a Kaggle input folder).
3. Run the notebook cells in order, top to bottom.

