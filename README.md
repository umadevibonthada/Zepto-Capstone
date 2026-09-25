# Zepto Data & AI Platform

Capstone project for the Certificate Program in Artificial Intelligence and Machine Learning.

## Modules

- Data Pipeline
- Analytics
- Support Assistant

## Project Structure

```text
data_pipeline/
analytics/
support_assistant/

```

# Module 2 — Titanic Analytics & Machine Learning Pipeline

df = sns.load_dataset("titanic")

Initial dataset size:

Rows: 891
Columns: 15
Task 1 — Dataset Profiling

The dataset was profiled using:

df.info()
df.describe()
df.shape

Initial missing values were found in:

Column Missing Percentage
age 19.87%
embarked 0.22%
deck 77.22%
embark_town 0.22%
==========================
Task 2 — Missing Value Handling

The following threshold rule was used:

Less than 5% missing → drop affected rows
5%–30% missing → impute missing values
Very high missing values → drop the column or treat missing as a separate category, with justification
Cleaning decisions
age → 19.87% missing → median imputation.
embarked → 0.22% missing → drop the affected rows.
embark_town → 0.22% missing → drop the affected rows.
deck → 77.22% missing → drop the column.
Justification for dropping deck

The deck column had 77.22% missing values. Because the proportion of missing values was very high, imputing the missing deck values would be unreliable. Therefore, the deck column was removed from the cleaned dataset.

Result after cleaning
Rows before cleaning: 891
Rows after cleaning: 889
Columns before cleaning: 15
Columns after cleaning: 14
Missing values after cleaning: 0

Fare is right-skewed because the mean (32.10) is greater than the median (14.45), and the median is greater than the mode (8.05). The high-value fares pull the mean toward the right side of the distribution.

## Multivariate Analysis

Chart 1 — Survival Rate by Sex and Passenger Class
Survival rates varied substantially by both sex and passenger class.
Female passengers had higher survival rates than male passengers within each passenger class.
First-class passengers generally had higher survival rates than passengers in lower classes.
Chart 2 — Age vs Fare by Survival
The scatter plot shows the relationship between passenger age and fare, separated by survival status.
Fare values are concentrated at lower levels, while a smaller number of passengers paid substantially higher fares.
Survival is therefore examined jointly with age and fare rather than considering either variable alone.
Chart 3 — Age Distribution by Survival and Sex
The age distributions differ across survival status and sex groups.
Female passengers who survived generally show a different age distribution from female passengers who did not survive, while male groups also show differences.
The chart helps examine how age, sex, and survival interact.
Chart 4 — Survival Rate by Embarked Port and Passenger Class
Survival rates vary across both embarkation port and passenger class.
Passenger class shows a noticeable relationship with survival within the different embarkation groups.
Combining embarkation and passenger class provides more detail than examining either variable separately.

Baseline: Precision 0.7813, Recall 0.7353, F1 0.7576
Class-weight balanced: Precision 0.7391, Recall 0.7500, F1 0.7445
SMOTE: Precision 0.7460, Recall 0.6912, F1 0.7176
The balanced model slightly increased recall, while the baseline had the highest F1 score. SMOTE produced the lowest recall and F1 among the three approaches.

The regression model achieved an R² of 0.3468 and an adjusted R² of 0.3120, indicating that the selected passenger features explain a limited portion of fare variation. The RMSE of 41.75 is higher than the MAE of 21.14, suggesting that some predictions have relatively large errors. The residual

plot should be checked for whether the spread of residuals changes across predicted fare values; a widening spread would indicate heteroscedasticity

madule 3

# Zepto Support Assistant

## Overview

This module implements an offline Retrieval-Augmented Generation (RAG) support assistant for Zepto policies.

The required graded baseline uses `MOCK_LLM=1` by default, so no external LLM API, API key, or network call is required.

## Architecture

The pipeline follows:

Ingestion → Embedding → Retrieval → Generation

### 1. Ingestion

Eight Zepto policy documents are stored in the `docs/` directory.

The `ingest.py` script loads all eight documents and stores them as individual chunks.

### 2. Embedding

The `sentence-transformers` library uses the `all-MiniLM-L6-v2` model to generate local embeddings.

The embeddings are stored in a ChromaDB collection named:

`zepto_policies`

### 3. Retrieval

The `retrieve_and_answer` LangGraph node embeds the incoming policy question and retrieves the top 3 most similar documents from ChromaDB using cosine similarity.

### 4. Generation

For the required offline mock mode, the retrieved top chunk is used to generate:

`Based on the retrieved context: ...`

For general questions, the `direct_answer` node returns a fixed response.

The `MOCK_LLM` environment variable controls only the generation/classification LLM branches. The embedding and ChromaDB retrieval continue to run locally.

## LangGraph Flow

```text
                    ┌────────────────────┐
                    │    User /ask       │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │  classify_intent   │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │   Conditional      │
                    │      Router        │
                    └──────┬───────┬─────┘
                           │       │
             policy_question       general_question
                           │       │
                           ↓       ↓
              ┌────────────────┐  ┌────────────────┐
              │ retrieve_and   │  │ direct_answer  │
              │ answer         │  │                │
              └───────┬────────┘  └───────┬────────┘
                      │                   │
                      └─────────┬─────────┘
                                ↓
                       Pydantic Response
```
