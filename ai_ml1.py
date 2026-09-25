from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# LOAD DATASET — ONLY NETWORK/CACHE LOAD
# ============================================================

df = sns.load_dataset(
    "titanic"
)


# ============================================================
# SAVE OFFLINE FALLBACK
# ============================================================

df.to_csv(
    BASE_DIR / "titanic.csv",
    index=False
)


# ============================================================
# BASIC PROFILE
# ============================================================

print("\n========== INFO ==========")

df.info()


print("\n========== DESCRIBE ==========")

print(
    df.describe(
        include="all"
    )
)


print("\n========== SHAPE ==========")

print(
    df.shape
)


# ============================================================
# MISSING VALUES
# ============================================================

missing_percent = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)

missing_percent = missing_percent[
    missing_percent > 0
]

print(
    "\n========== MISSING % =========="
)

print(
    missing_percent
)


missing_percent.to_csv(
    OUTPUT_DIR /
    "missing_percentages.csv"
)


# ============================================================
# CLEANING
# ============================================================

clean_df = df.copy()


# ---- under 5% ----

for column in clean_df.columns:

    missing_rate = (
        clean_df[column]
        .isna()
        .mean()
        * 100
    )

    if (
        missing_rate > 0
        and missing_rate < 5
    ):

        clean_df = clean_df.dropna(
            subset=[column]
        )


# ---- 5% to 30% ----

if "age" in clean_df.columns:

    age_missing = (
        clean_df["age"]
        .isna()
        .mean()
        * 100
    )

    if 5 <= age_missing <= 30:

        clean_df["age"] = (
            clean_df["age"]
            .fillna(
                clean_df["age"].median()
            )
        )


# ---- high missingness ----

if "deck" in clean_df.columns:

    clean_df["deck"] = (
        clean_df["deck"]
        .astype("string")
        .fillna("Missing")
    )


# Other small categorical gaps

for column in ["embarked"]:

    if column in clean_df.columns:

        clean_df[column] = (
            clean_df[column]
            .astype("string")
            .fillna("Missing")
        )


# ============================================================
# REMOVE REDUNDANT TARGET-LEAKING / DERIVED COLUMNS
# ============================================================

clean_df = clean_df.drop(
    columns=[
        "alive",
        "class",
        "who",
        "adult_male",
        "alone"
    ],
    errors="ignore"
)


clean_df.to_csv(
    BASE_DIR / "cleaned_titanic.csv",
    index=False
)


# ============================================================
# IQR FUNCTION
# ============================================================

def calculate_iqr_outliers(series):

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    mask = (
        (series < lower)
        |
        (series > upper)
    )

    return (
        int(mask.sum()),
        lower,
        upper
    )


# ============================================================
# AGE ANALYSIS
# ============================================================

age_outliers, age_lower, age_upper = (
    calculate_iqr_outliers(
        clean_df["age"]
    )
)


plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    clean_df["age"],
    kde=True
)

plt.title(
    "Age Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_hist.png"
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    x=clean_df["age"]
)

plt.title(
    "Age Box Plot"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_box.png"
)

plt.close()


# ============================================================
# FARE ANALYSIS
# ============================================================

fare_outliers, fare_lower, fare_upper = (
    calculate_iqr_outliers(
        clean_df["fare"]
    )
)


plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    clean_df["fare"],
    kde=True
)

plt.title(
    "Fare Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_hist.png"
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    x=clean_df["fare"]
)

plt.title(
    "Fare Box Plot"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_box.png"
)

plt.close()


# ============================================================
# FARE MEAN MEDIAN MODE
# ============================================================

fare_mean = clean_df["fare"].mean()

fare_median = clean_df["fare"].median()

fare_mode = clean_df["fare"].mode().iloc[0]


print("\n========== FARE ==========")

print(
    "Mean:",
    fare_mean
)

print(
    "Median:",
    fare_median
)

print(
    "Mode:",
    fare_mode
)

print(
    "Fare outliers:",
    fare_outliers
)

print(
    "Age outliers:",
    age_outliers
)


# ============================================================
# SURVIVAL BY SEX
# ============================================================

survival_by_sex = (
    clean_df
    .groupby("sex")["survived"]
    .mean()
)

print(
    "\n========== SURVIVAL BY SEX =========="
)

print(
    survival_by_sex
)


# ============================================================
# SURVIVAL BY CLASS
# ============================================================

survival_by_class = (
    clean_df
    .groupby("pclass")["survived"]
    .mean()
)

print(
    "\n========== SURVIVAL BY PCLASS =========="
)

print(
    survival_by_class
)


# ============================================================
# BOOLEAN MASKING
# ============================================================

female_first = clean_df[
    (clean_df["sex"] == "female")
    &
    (clean_df["pclass"] == 1)
]

male_third = clean_df[
    (clean_df["sex"] == "male")
    &
    (clean_df["pclass"] == 3)
]


print(
    "\nFemale + First Class:",
    female_first["survived"].mean()
)

print(
    "Male + Third Class:",
    male_third["survived"].mean()
)


# ============================================================
# SEX + CLASS
# ============================================================

survival_by_sex_class = (
    clean_df
    .groupby(
        ["sex", "pclass"]
    )["survived"]
    .mean()
)

print(
    "\n========== SEX + CLASS =========="
)

print(
    survival_by_sex_class
)


# ============================================================
# CORRELATION MATRIX
# EXACT SIX REQUIRED COLUMNS
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = (
    clean_df[
        correlation_columns
    ]
    .corr()
)


print(
    "\n========== CORRELATION =========="
)

print(
    correlation_matrix
)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Titanic Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "correlation_heatmap.png"
)

plt.close()


# ============================================================
# TWO STRONGEST CORRELATIONS
# ============================================================

pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        col1 = correlation_columns[i]

        col2 = correlation_columns[j]

        value = correlation_matrix.loc[
            col1,
            col2
        ]

        pairs.append(
            (
                col1,
                col2,
                value,
                abs(value)
            )
        )


pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)


print(
    "\n========== TWO STRONGEST CORRELATIONS =========="
)

for pair in pairs[:2]:

    print(
        pair
    )


# ============================================================
# MULTIVARIATE CHART 1
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=clean_df,
    x="sex",
    y="survived",
    hue="pclass"
)

plt.title(
    "Survival by Sex and Passenger Class"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "survival_sex_pclass.png"
)

plt.close()


# ============================================================
# MULTIVARIATE CHART 2
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=clean_df,
    x="survived",
    y="age"
)

plt.title(
    "Age Distribution by Survival"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "age_survival.png"
)

plt.close()


# ============================================================
# MULTIVARIATE CHART 3
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(
    data=clean_df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title(
    "Age vs Fare by Survival"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "age_fare_survival.png"
)

plt.close()


# ============================================================
# MULTIVARIATE CHART 4
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=clean_df,
    x="embarked",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival by Embarkation Port and Sex"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "embarked_sex_survival.png"
)

plt.close()


# ============================================================
# EXPLORATORY STANDARDIZATION
# ============================================================

scaler = StandardScaler()

standardized = scaler.fit_transform(
    clean_df[
        ["age", "fare"]
    ]
)

standardized_df = pd.DataFrame(
    standardized,
    columns=[
        "age",
        "fare"
    ]
)


print(
    "\n========== BEFORE STANDARDIZATION =========="
)

print(
    clean_df[
        ["age", "fare"]
    ].agg(
        ["mean", "std"]
    )
)


print(
    "\n========== AFTER STANDARDIZATION =========="
)

print(
    standardized_df.agg(
        ["mean", "std"]
    )
)


# ============================================================
# REPORT
# ============================================================

with open(
    OUTPUT_DIR / "eda_report.md",
    "w",
    encoding="utf-8"
) as file:

    file.write("# EDA Report\n\n")

    file.write(
        f"Raw dataset shape: {df.shape}\n\n"
    )

    file.write(
        f"Cleaned dataset shape: {clean_df.shape}\n\n"
    )

    file.write(
        "## Missing-value percentages\n\n"
    )

    file.write(
        missing_percent.to_string()
    )

    file.write("\n\n")

    file.write(
        "## Outliers\n\n"
    )

    file.write(
        f"- Age IQR outliers: {age_outliers}\n"
    )

    file.write(
        f"- Fare IQR outliers: {fare_outliers}\n\n"
    )

    file.write(
        "## Fare distribution\n\n"
    )

    if (
        fare_mean
        >
        fare_median
        >
        fare_mode
    ):

        skew = "right-skewed"

    elif (
        fare_mean
        <
        fare_median
        <
        fare_mode
    ):

        skew = "left-skewed"

    else:

        skew = "approximately symmetric/mixed"

    file.write(
        f"Mean = {fare_mean:.4f}\n\n"
    )

    file.write(
        f"Median = {fare_median:.4f}\n\n"
    )

    file.write(
        f"Mode = {fare_mode:.4f}\n\n"
    )

    file.write(
        f"The fare distribution is {skew}.\n\n"
    )

    file.write(
        "## Survival by sex\n\n"
    )

    file.write(
        survival_by_sex.to_string()
    )

    file.write("\n\n")

    file.write(
        "## Survival by passenger class\n\n"
    )

    file.write(
        survival_by_class.to_string()
    )

    file.write("\n\n")

    file.write(
        "## Survival by sex and class\n\n"
    )

    file.write(
        survival_by_sex_class.to_string()
    )

    file.write("\n\n")

    file.write(
        "## Strongest correlations\n\n"
    )

    for pair in pairs[:2]:

        file.write(
            f"- {pair[0]} vs {pair[1]}: "
            f"{pair[2]:.4f}\n"
        )

    file.write("\n")

    file.write(
        "## Chart interpretations\n\n"
    )

    file.write(
        "### 1. Survival by Sex and Class\n\n"
        "This chart compares survival across sex and passenger class. "
        "The grouping makes it possible to observe the joint relationship "
        "between these two categorical variables and survival.\n\n"
    )

    file.write(
        "### 2. Age Distribution by Survival\n\n"
        "The box plot compares the age distributions of survivors and "
        "non-survivors. Differences in the median and spread indicate "
        "whether age is associated with survival.\n\n"
    )

    file.write(
        "### 3. Age versus Fare\n\n"
        "The scatter plot examines the relationship between age and fare "
        "while using survival as the grouping variable. This provides a "
        "multivariate view that cannot be obtained from either variable alone.\n\n"
    )

    file.write(
        "### 4. Embarkation Port and Sex\n\n"
        "This chart compares survival rates across embarkation ports "
        "while separating the observations by sex. It provides another "
        "view of how demographic and travel-related variables interact "
        "with survival.\n\n"
    )

print(
    "\nEDA completed successfully."
)
