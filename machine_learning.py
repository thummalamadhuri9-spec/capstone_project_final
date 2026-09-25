from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    exist_ok=True
)

DATA_FILE = BASE_DIR / "cleaned_titanic.csv"


# ============================================================
# LOAD COMMITTED CSV
# IMPORTANT:
# No sns.load_dataset() here.
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# CLASSIFICATION DATA
# ============================================================

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]


X = df[features]

y = df[target]


# ============================================================
# CLASS BALANCE
# ============================================================

class_balance = (
    y.value_counts(
        normalize=True
    )
    .sort_index()
)

print(
    "\n========== CLASS BALANCE =========="
)

print(
    class_balance
)


# ============================================================
# STRATIFIED SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTrain:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Decision Tree":

        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
}


results = []

roc_data = {}


# ============================================================
# TRAIN THREE CLASSIFIERS
# ============================================================

fitted_pipelines = {}


for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auc": auc
    })


    fitted_pipelines[
        model_name
    ] = pipeline


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        OUTPUT_DIR /
        f"confusion_matrix_{filename}.png"
    )

    plt.close()


    # --------------------------------------------------------
    # ROC DATA
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_data[
        model_name
    ] = (
        fpr,
        tpr,
        auc
    )


# ============================================================
# CLASSIFIER RESULTS
# ============================================================

classification_results = pd.DataFrame(
    results
)

classification_results.to_csv(
    BASE_DIR /
    "model_results.csv",
    index=False
)


print(
    "\n========== CLASSIFICATION RESULTS =========="
)

print(
    classification_results
)


# ============================================================
# ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 6)
)

for model_name, (
    fpr,
    tpr,
    auc
) in roc_data.items():

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} AUC={auc:.3f}"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "roc_curves.png"
)

plt.close()


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

tree_pipeline = fitted_pipelines[
    "Decision Tree"
]

tree_model = tree_pipeline.named_steps[
    "model"
]

tree_preprocessor = (
    tree_pipeline.named_steps[
        "preprocessor"
    ]
)

feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)


plt.figure(
    figsize=(24, 12)
)

plot_tree(

    tree_model,

    feature_names=feature_names,

    class_names=[
        "Not Survived",
        "Survived"
    ],

    filled=True,

    max_depth=3,

    fontsize=8
)

plt.title(
    "Decision Tree"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "decision_tree.png"
)

plt.close()


# ============================================================
# IMBALANCE COMPARISON
# ============================================================

imbalance_results = []


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


baseline_model.fit(
    X_train,
    y_train
)


baseline_pred = baseline_model.predict(
    X_test
)


imbalance_results.append({

    "strategy": "Baseline",

    "precision": precision_score(
        y_test,
        baseline_pred,
        zero_division=0
    ),

    "recall": recall_score(
        y_test,
        baseline_pred,
        zero_division=0
    ),

    "f1": f1_score(
        y_test,
        baseline_pred,
        zero_division=0
    )
})


# ------------------------------------------------------------
# CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


balanced_model.fit(
    X_train,
    y_train
)


balanced_pred = balanced_model.predict(
    X_test
)


imbalance_results.append({

    "strategy":
        "class_weight=balanced",

    "precision":
        precision_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),

    "recall":
        recall_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),

    "f1":
        f1_score(
            y_test,
            balanced_pred,
            zero_division=0
        )
})


# ------------------------------------------------------------
# SMOTE
# IMPORTANT:
# SMOTE is INSIDE the pipeline.
# Therefore it is applied only to training folds.
# ------------------------------------------------------------

smote_model = ImbPipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


smote_model.fit(
    X_train,
    y_train
)


smote_pred = smote_model.predict(
    X_test
)


imbalance_results.append({

    "strategy": "SMOTE",

    "precision":
        precision_score(
            y_test,
            smote_pred,
            zero_division=0
        ),

    "recall":
        recall_score(
            y_test,
            smote_pred,
            zero_division=0
        ),

    "f1":
        f1_score(
            y_test,
            smote_pred,
            zero_division=0
        )
})


imbalance_df = pd.DataFrame(
    imbalance_results
)


imbalance_df.to_csv(
    BASE_DIR /
    "imbalance_comparison.csv",
    index=False
)


print(
    "\n========== IMBALANCE COMPARISON =========="
)

print(
    imbalance_df
)


# ============================================================
# RANDOM FOREST GRID SEARCH
# ============================================================

rf_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                random_state=42,
                oob_score=True,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(

    estimator=rf_pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


best_rf_pipeline = (
    grid_search.best_estimator_
)


best_rf_model = (
    best_rf_pipeline
    .named_steps["model"]
)


print(
    "\n========== GRID SEARCH =========="
)

print(
    "Best parameters:"
)

print(
    grid_search.best_params_
)

print(
    "Best CV score:",
    grid_search.best_score_
)

print(
    "OOB score:",
    best_rf_model.oob_score_
)


with open(
    BASE_DIR /
    "grid_search_results.txt",
    "w"
) as file:

    file.write(
        f"Best Parameters: "
        f"{grid_search.best_params_}\n"
    )

    file.write(
        f"Best CV F1: "
        f"{grid_search.best_score_}\n"
    )

    file.write(
        f"OOB Score: "
        f"{best_rf_model.oob_score_}\n"
    )


# ============================================================
# REGRESSION
# Predict FARE
# ============================================================

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]


X_reg = df[
    regression_features
]

y_reg = df[
    "fare"
]


X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(

        X_reg,
        y_reg,

        test_size=0.20,

        random_state=42
    )
)


reg_numeric = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]


reg_categorical = [
    "sex",
    "embarked"
]


reg_preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",

            Pipeline(
                steps=[

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),

            reg_numeric
        ),

        (
            "categorical",

            Pipeline(
                steps=[

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),

            reg_categorical
        )
    ]
)


regression_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            reg_preprocessor
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


fare_prediction = (
    regression_pipeline.predict(
        X_reg_test
    )
)


mae = mean_absolute_error(
    y_reg_test,
    fare_prediction
)


rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        fare_prediction
    )
)


r2 = r2_score(
    y_reg_test,
    fare_prediction
)


n = len(y_reg_test)

p = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(
        X_reg_test
    )
    .shape[1]
)


adjusted_r2 = (
    1
    -
    (
        (1 - r2)
        *
        (n - 1)
        /
        (n - p - 1)
    )
)


print(
    "\n========== REGRESSION =========="
)

print(
    "MAE:",
    mae
)

print(
    "RMSE:",
    rmse
)

print(
    "R2:",
    r2
)

print(
    "Adjusted R2:",
    adjusted_r2
)


# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test
    -
    fare_prediction
)


plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    fare_prediction,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Fare Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "residual_plot.png"
)

plt.close()


# ============================================================
# REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame({

    "MAE": [mae],

    "RMSE": [rmse],

    "R2": [r2],

    "Adjusted_R2": [
        adjusted_r2
    ]
})


regression_results.to_csv(
    BASE_DIR /
    "regression_results.csv",
    index=False
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

comparison = classification_results.copy()

comparison[
    "MAE"
] = np.nan

comparison[
    "RMSE"
] = np.nan

comparison[
    "R2"
] = np.nan

comparison[
    "Adjusted_R2"
] = np.nan


comparison.to_csv(
    BASE_DIR /
    "model_comparison.csv",
    index=False
)


print(
    "\n========== MODEL COMPARISON =========="
)

print(
    comparison
)


# ============================================================
# SAVE COMPLETE PIPELINE
# ============================================================

joblib.dump(

    best_rf_pipeline,

    BASE_DIR /
    "random_forest_pipeline.joblib"
)


# ============================================================
# RELOAD PIPELINE
# ============================================================

loaded_pipeline = joblib.load(

    BASE_DIR /
    "random_forest_pipeline.joblib"
)


raw_sample = X_test.iloc[
    :5
]


reloaded_predictions = (
    loaded_pipeline.predict(
        raw_sample
    )
)


print(
    "\n========== RELOADED PIPELINE =========="
)

print(
    reloaded_predictions
)


# ============================================================
# FINAL README DATA
# ============================================================

with open(
    BASE_DIR /
    "modeling_report.md",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "# Modeling Report\n\n"
    )

    file.write(
        "## Stratification\n\n"
        "The train/test split uses stratification so that the "
        "survived/not-survived class proportions remain approximately "
        "consistent between training and test data.\n\n"
    )

    file.write(
        "## Preprocessing\n\n"
        "Numeric features use median imputation and StandardScaler. "
        "Categorical features use most-frequent imputation and one-hot encoding. "
        "All preprocessing is contained in the scikit-learn pipeline, so fitting "
        "occurs only on the training data.\n\n"
    )

    file.write(
        "## Imbalance\n\n"
    )

    file.write(
        imbalance_df.to_markdown(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "The three strategies are compared using precision, recall and F1. "
        "The choice of strategy should be based on the metric priorities of "
        "the application rather than class balance alone.\n\n"
    )

    file.write(
        "## Random Forest Grid Search\n\n"
    )

    file.write(
        f"Best parameters: "
        f"`{grid_search.best_params_}`\n\n"
    )

    file.write(
        f"Best cross-validation F1: "
        f"{grid_search.best_score_:.4f}\n\n"
    )

    file.write(
        f"OOB score: "
        f"{best_rf_model.oob_score_:.4f}\n\n"
    )

    file.write(
        "## Regression\n\n"
    )

    file.write(
        f"- MAE: {mae:.4f}\n"
    )

    file.write(
        f"- RMSE: {rmse:.4f}\n"
    )

    file.write(
        f"- R²: {r2:.4f}\n"
    )

    file.write(
        f"- Adjusted R²: {adjusted_r2:.4f}\n\n"
    )

    file.write(
        "The residual plot should be inspected for whether the spread of "
        "residuals changes systematically with predicted fare. A funnel-shaped "
        "or otherwise systematic spread would indicate heteroscedasticity; "
        "a relatively uniform random cloud around zero would not.\n\n"
    )

    file.write(
        "## Classifier Metrics\n\n"
    )

    file.write(
        classification_results.to_markdown(
            index=False
        )
    )

print(
    "\nModule 2 modeling completed successfully."
)
