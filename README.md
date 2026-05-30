# Hotel Reservation Cancellation Prediction

A machine-learning project that predicts whether a hotel booking will be **canceled** or **honored**, based on reservation details available at booking time. The goal is to give a hotel an early signal of at-risk reservations so it can act before the cancellation happens (overbooking strategy, deposits, targeted offers, reminders).

---

## Table of Contents

- [Problem](#problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Key EDA Findings](#key-eda-findings)
- [Preprocessing](#preprocessing)
- [Modeling & Results](#modeling--results)
- [Model Interpretation](#model-interpretation)
- [Business Recommendations](#business-recommendations)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Authors](#authors)

---

## Problem

Cancellations cost hotels revenue and make capacity planning hard. This project frames the question as a **binary classification** task:

> Given the information known at the time of booking, will this reservation be canceled?

`booking_status` is the target, encoded as `Canceled = 1`, `Not_Canceled = 0`.

## Dataset

[Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) (Kaggle).

- **36,275** bookings, **19** columns (`Booking_ID` + 17 features + target).
- **No missing values, no duplicates.**
- **Class balance:** ~67.2% not canceled, ~32.8% canceled.
- A copy of the data ships with the repo at `data/Hotel Reservations.csv`.

<details>
<summary><b>Feature reference</b></summary>

| Feature | Description |
| --- | --- |
| `no_of_adults`, `no_of_children` | Number of adults / children |
| `no_of_weekend_nights`, `no_of_week_nights` | Weekend (Sat/Sun) and weekday nights booked |
| `type_of_meal_plan` | Meal plan selected |
| `required_car_parking_space` | Whether a parking space was requested (0/1) |
| `room_type_reserved` | Reserved room type (encoded categories) |
| `lead_time` | Days between booking and arrival |
| `arrival_year`, `arrival_month`, `arrival_date` | Arrival date components |
| `market_segment_type` | Booking channel / segment (Online, Offline, Corporate, Aviation, Complementary) |
| `repeated_guest` | Whether the guest is a returning customer (0/1) |
| `no_of_previous_cancellations` | Prior cancellations by the guest |
| `no_of_previous_bookings_not_canceled` | Prior honored bookings by the guest |
| `avg_price_per_room` | Average price per room (in the dataset's currency) |
| `no_of_special_requests` | Number of special requests made |
| `booking_status` | **Target** — `Canceled` / `Not_Canceled` |

</details>

## Project Structure

```
hotel-model/
├── data/
│   └── Hotel Reservations.csv     # raw dataset (input)
├── notebooks/
│   ├── 01_eda.ipynb               # exploratory data analysis
│   ├── 02_preprocessing.ipynb     # cleaning, feature engineering, split, encoding
│   ├── 03_modeling.ipynb          # train, tune, compare, evaluate models
│   └── 04_interpretation.ipynb    # feature importance & business insights
├── requirements.txt               # Python dependencies
└── README.md
```

Running the notebooks generates two artifacts (not tracked in git):

- `data/processed_data.pkl` — train/val/test splits + preprocessed matrices (written by `02`).
- `data/best_model.pkl` — the tuned final model (written by `03`).

## Workflow

The project is organized as a four-stage pipeline, one notebook per stage:

1. **EDA** (`01`) — understand the data, check quality, form hypotheses.
2. **Preprocessing** (`02`) — clean, engineer features, split, encode, persist.
3. **Modeling** (`03`) — train several classifiers, tune the best, evaluate on a held-out test set.
4. **Interpretation** (`04`) — explain *why* the model predicts cancellations and translate it into business actions.

## Key EDA Findings

- **Lead time matters most.** The earlier a reservation is made, the more likely it is to be canceled.
- **Returning guests rarely cancel:** only **1.7%** of repeated guests cancel, vs **33.6%** of new guests.
- **Booking channel drives risk** — cancellation rate by `market_segment_type`:

  | Segment | Cancellation rate |
  | --- | ---: |
  | Online | 36.5% |
  | Offline | 30.0% |
  | Aviation | 29.6% |
  | Corporate | 10.9% |
  | Complementary | 0.0% |

- **Free rooms (`avg_price_per_room == 0`)** are almost never canceled → motivated the engineered `is_free_room` flag.
- **Zero-night ("day use") bookings** cancel only **2.56%** of the time vs **32.83%** for normal stays → motivated the engineered `is_zero_nights` flag.
- Data is clean: **no missing values or duplicates**; price and lead time contain many high-end outliers (kept intentionally).

## Preprocessing

Implemented in `02_preprocessing.ipynb`:

- **Drop** the `Booking_ID` identifier (no predictive value).
- **Encode target:** `Canceled → 1`, `Not_Canceled → 0`.
- **Feature engineering:**
  - `is_free_room` — 1 when `avg_price_per_room == 0`.
  - `is_zero_nights` — 1 when total booked nights (`week + weekend`) is 0.
- **Stratified split** into train / validation / test at **80 / 10 / 10** (stratified on the target, `random_state=123`).
- **Preprocessing pipeline** (`ColumnTransformer`):
  - Numeric features → `StandardScaler`.
  - Categorical features (`type_of_meal_plan`, `room_type_reserved`, `market_segment_type`) → `OrdinalEncoder` (with `handle_unknown='use_encoded_value'`).
- Splits and transformed matrices are persisted with `joblib` to `data/processed_data.pkl`.

## Modeling & Results

Implemented in `03_modeling.ipynb`. Four baseline classifiers were trained, then the strongest (Random Forest) was tuned with `RandomizedSearchCV` (100 iterations, 5-fold CV, optimizing **F1**).

**Validation-set comparison:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8131 | 0.7502 | 0.6442 | 0.6932 | 0.8666 |
| Decision Tree | 0.8782 | 0.8131 | 0.8158 | 0.8144 | 0.8641 |
| Random Forest | 0.9135 | 0.9003 | 0.8276 | 0.8624 | 0.9606 |
| SVC | 0.8396 | 0.8056 | 0.6728 | 0.7333 | 0.8943 |
| **Tuned Random Forest** | **0.9143** | 0.8991 | **0.8318** | **0.8641** | **0.9616** |

**Best model — Tuned Random Forest.** Best hyperparameters:

```python
{ "n_estimators": 300, "max_depth": 25, "min_samples_split": 3, "min_samples_leaf": 1 }
```

**Held-out test-set performance** (final, unbiased estimate):

| Metric | Score |
| --- | ---: |
| Accuracy | 0.9016 |
| Precision | 0.8694 |
| Recall | 0.8234 |
| F1-Score | 0.8458 |
| ROC-AUC | 0.9589 |

The model correctly flags ~82% of real cancellations while keeping precision high (~87%), with strong overall separability (ROC-AUC ≈ 0.96).

## Model Interpretation

From the Random Forest feature importances (`04_interpretation.ipynb`), the strongest predictors are:

1. **`lead_time`** — risk rises steadily the further out the booking is made.
2. **`avg_price_per_room`** — higher-priced rooms are canceled more often (guests shop for cheaper alternatives).
3. **`no_of_special_requests`** — more requests ⇒ more committed guests ⇒ fewer cancellations.
4. **`arrival_month` / `arrival_date`** — seasonality; cancellations peak in spring/summer months.
5. **`market_segment_type`** — segments behave very differently (Online/Offline riskiest, Corporate/Complementary safest).
6. **Length of stay** (`no_of_week_nights` + `no_of_weekend_nights`).

Segment-level analysis showed the effects interact: price sensitivity is strongest in the Online/Offline segments, while Corporate is comparatively stable.

## Business Recommendations

- **Manage long-lead bookings actively** — reminders and partial deposits for reservations made far in advance.
- **Protect high-value bookings** — offer perks/benefits on expensive rooms to reduce cancellations.
- **Encourage special requests** — prompting guests to add preferences after booking correlates with lower cancellation risk.
- **Segment-specific policies** — treat Online/Offline (high risk) differently from Corporate/Complementary.
- **Watch seasonality** — pay particular attention to spring/summer cancellations, especially in the Corporate segment.

## Getting Started

**Prerequisites:** Python 3.10+ and Jupyter.

```bash
# 1. Clone
git clone https://github.com/szczuror/hotel-model.git
cd hotel-model

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Run the notebooks in order** — each stage consumes the previous stage's output:

```bash
jupyter notebook   # or: jupyter lab
```

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`  → writes `data/processed_data.pkl`
3. `notebooks/03_modeling.ipynb`        → writes `data/best_model.pkl`
4. `notebooks/04_interpretation.ipynb`

> Notebooks 03 and 04 load the `.pkl` files produced earlier, so run 02 before 03, and 03 before 04.

## Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — pipelines, preprocessing, models, `RandomizedSearchCV`, metrics
- **matplotlib** / **seaborn** — visualization
- **joblib** — artifact persistence
- **Jupyter** — notebooks

## Authors

Team project by **Igor Kowalik** and **Wiktoria Postek**.

Dataset: *Hotel Reservations Classification Dataset*, available on [Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset).
