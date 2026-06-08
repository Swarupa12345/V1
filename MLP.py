# cl_cd_analysis.py

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

DATA_PATH        = "DRDL_aero_data_final.csv"
SCALER_PATH      = "feature_scaler.pkl"
CL_MODEL_PATH    = "cl_xgb.model"
CD_MODEL_PATH    = "cd_xgb.model"

MACH_LIST = [0.5, 0.8]          # Mach numbers to analyse
ALT_LIST  = [0, 6000]          # Altitudes in metres


df = pd.read_csv(DATA_PATH)
df = df.apply(pd.to_numeric, errors="coerce")   # non‑numeric → 

FEATURES = [
    "nose length", "body_length", "wing LE", "root chord", "tip chord",
    "semi-span", "root th", "tip th", "wing sweep", "tail LE",
    "root chord.1", "tip chord.1", "semi-span.1", "root th.1",
    "tip th.1", "MACH", "ALPHA", "ALT",
]
FEATURES = [c for c in FEATURES if c in df.columns]

X = df[FEATURES]


scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)


def load_model(path: str) -> xgb.XGBRegressor:
    """Load an XGBoost model; raise a clear error if missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model

cl_model = load_model(CL_MODEL_PATH)
cd_model = load_model(CD_MODEL_PATH)


df["CL_pred"] = cl_model.predict(X_scaled)
df["CD_pred"] = cd_model.predict(X_scaled)


df["CL_CD_actual"] = df["CL"] / df["CD"]
df["CL_CD_pred"]   = df["CL_pred"] / df["CD_pred"]


def assign_weights(row: pd.Series, scheme: str) -> float:
    """
    Return a weight for a single flight point.

    * scheme == "uniform"  → constant weight = 1.0
    * scheme == "variable" → Mach factor × α factor
    """
    # ---- uniform ----
    if scheme == "uniform":
        return 1.0

    # ---- variable ----
    w = 1.0

    # Mach factor
    if row["MACH"] in {0.6, 0.7, 0.8}:
        w *= 2.0          # higher weight for high‑Mach regime
    elif row["MACH"] == 0.2:
        w *= 0.5          # lower weight for low‑Mach regime
    # Mach = 0.5 (our data) → factor stays 1.0

    # α factor
    alpha = int(row["ALPHA"])
    if alpha in {4, 6}:
        w *= 2.0          # boost for the critical angles
    elif alpha > 10:
        w *= 0.5          # de‑emphasise large angles
    # otherwise factor stays 1.0

    return w

# create weight columns
df["weight_uniform"]  = df.apply(assign_weights, axis=1, scheme="uniform")
df["weight_variable"] = df.apply(assign_weights, axis=1, scheme="variable")

# weighted prediction (only the variable scheme is needed for the plot)
df["CL_CD_pred_weighted"] = df["CL_CD_pred"] * df["weight_variable"]


def filter_data(mach: float, altitude: int) -> pd.DataFrame:
    """Return rows that match a given Mach number and altitude."""
    mask = (np.isclose(df["MACH"], mach)) & (df["ALT"] == altitude)
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No data for Mach={mach}, ALT={altitude}")
    return subset.sort_values("ALPHA")

def plot_cl_cd(subset: pd.DataFrame, mach: float, altitude: int) -> None:
    """Plot actual CL/CD and the three prediction curves."""
    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    # Actual (blue)
    sns.lineplot(
        data=subset,
        x="ALPHA",
        y="CL_CD_actual",
        label="Actual",
        color="tab:blue",
        marker="o",
        linestyle="-",
    )
    # Raw prediction (magenta)
    sns.lineplot(
        data=subset,
        x="ALPHA",
        y="CL_CD_pred",
        label="Pred (raw)",
        color="tab:pink",
        marker="s",
        linestyle="-",
    )
    # Uniform‑weighted prediction – identical to raw because weight=1
    sns.lineplot(
        data=subset,
        x="ALPHA",
        y="CL_CD_pred",
        label="Pred (uniform weight)",
        color="tab:green",
        marker="^",
        linestyle="--",
    )
    # Variable‑weighted prediction – orange
    sns.lineplot(
        data=subset,
        x="ALPHA",
        y="CL_CD_pred_weighted",
        label="Pred (variable weight)",
        color="tab:orange",
        marker="D",
        linestyle="-.",
    )

    plt.title(f"CL/CD vs. α  (Mach = {mach}, Alt = {altitude} m)")
    plt.xlabel("Incidence angle α (°)")
    plt.ylabel("Lift‑to‑Drag ratio (CL/CD)")
    plt.legend(title="Series")
    plt.tight_layout()
    plt.show()


for mach in MACH_LIST:
    for alt in ALT_LIST:
        subset = filter_data(mach, alt)

        # Plot the four curves
        plot_cl_cd(subset, mach, alt)

        # Export CSV with all relevant columns
        export_cols = [
            "ALPHA",
            "CL", "CD",
            "CL_pred", "CD_pred",
            "CL_CD_actual", "CL_CD_pred",
            "weight_uniform", "weight_variable",
            "CL_CD_pred_weighted",
        ]
        out_file = f"results_M{mach}_ALT{alt}.csv"
        subset[export_cols].to_csv(out_file, index=False)
        print(f"Saved CSV → {out_file}")








# ------------------------------------------------------------
# 0️⃣  Imports
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# 1️⃣  File paths  (change if your files are elsewhere)
# ------------------------------------------------------------
DATA_PATH      = "DRDL_aero_data_final.csv"
SCALER_PATH    = "feature_scaler.pkl"
CL_MODEL_PATH  = "cl_xgb.model"
CD_MODEL_PATH  = "cd_xgb.model"

# ------------------------------------------------------------
# 2️⃣  Load data & saved objects
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Load the scaler and the two XGBoost models exactly as they were saved
scaler = joblib.load(SCALER_PATH)

cl_model = xgb.XGBRegressor()
cl_model.load_model(CL_MODEL_PATH)

cd_model = xgb.XGBRegressor()
cd_model.load_model(CD_MODEL_PATH)

# ------------------------------------------------------------
# 3️⃣  Feature list – must match the list used during training
# ------------------------------------------------------------
features = [
    "nose length", "body_length", "wing LE", "root chord", "tip chord",
    "semi-span", "root th", "tip th", "wing sweep", "tail LE",
    "root chord.1", "tip chord.1", "semi-span.1", "root th.1",
    "tip th.1", "MACH", "ALPHA", "ALT",
]

# Keep only columns that actually exist after numeric conversion
features = [c for c in features if c in df.columns]

# ------------------------------------------------------------
# 4️⃣  Build the feature matrix and scale it
# ------------------------------------------------------------
X = df[features].apply(pd.to_numeric, errors="coerce")
X_scaled = scaler.transform(X)          # same scaler as during training

# ------------------------------------------------------------
# 5️⃣  Predict CL and CD for every row
# ------------------------------------------------------------
df["CL_pred"] = cl_model.predict(X_scaled)
df["CD_pred"] = cd_model.predict(X_scaled)

# ------------------------------------------------------------
# 6️⃣  Compute lift‑to‑drag ratios (actual & predicted)
# ------------------------------------------------------------
# Simple formula: CL/CD = CL ÷ CD
df["CL_CD_actual"] = df["CL"] / df["CD"]
df["CL_CD_pred"]   = df["CL_pred"] / df["CD_pred"]

# ------------------------------------------------------------
# 7️⃣  Evaluate – R² between actual and predicted CL/CD
# ------------------------------------------------------------
# Drop rows where either ratio is NaN or infinite (e.g., CD = 0)
valid_mask = (
    df["CL_CD_actual"].notna() &
    df["CL_CD_pred"].notna() &
    np.isfinite(df["CL_CD_actual"]) &
    np.isfinite(df["CL_CD_pred"])
)
df_valid = df.loc[valid_mask]

r2 = r2_score(df_valid["CL_CD_actual"], df_valid["CL_CD_pred"])
print(f"R² (Predicted CL/CD vs. Actual CL/CD) = {r2:.4f}")

# ------------------------------------------------------------
# 8️⃣  Plot Predicted CL/CD vs. Actual CL/CD
# ------------------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Scatter of the two ratios
sns.scatterplot(
    data=df_valid,
    x="CL_CD_actual",
    y="CL_CD_pred",
    alpha=0.6,
    edgecolor=None,
)

# Add the 45° reference line (perfect prediction)
max_val = max(df_valid["CL_CD_actual"].max(), df_valid["CL_CD_pred"].max())
min_val = min(df_valid["CL_CD_actual"].min(), df_valid["CL_CD_pred"].min())
plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linestyle="--", linewidth=1,
         label="Ideal (y = x)")

plt.title("Predicted vs. Actual CL/CD")
plt.xlabel("Actual CL/CD")
plt.ylabel("Predicted CL/CD")
plt.legend()
plt.tight_layout()

# Save the figure (optional)
plt.savefig("predicted_vs_actual_cl_cd.png", dpi=300)

plt.show()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
file_path = "DRDL_aero_data_final.csv"
data = pd.read_csv(file_path)

# Remove constant & ID columns
constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
data_cleaned = data.drop(columns=constant_columns + ['CASEID'])
data_numeric = data_cleaned.apply(pd.to_numeric, errors='coerce')

selected_features = ["ALPHA","MACH","X-C.P.","nose length","semi-span","wing sweep"]
X = data_numeric[selected_features]
y = data_numeric[['CN','CM']]

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled   = x_scaler.transform(X_val)
X_test_scaled  = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled   = y_scaler.transform(y_val)
y_test_scaled  = y_scaler.transform(y_test)

# Model
epochs = 500
train_losses = []
val_losses = []

model = MLPRegressor(hidden_layer_sizes=(64,64), activation='relu',
                     solver='adam', max_iter=1, warm_start=True, random_state=42)

for epoch in range(epochs):
    model.partial_fit(X_train_scaled, y_train_scaled)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred   = model.predict(X_val_scaled)

    train_loss = mean_squared_error(y_train_scaled, y_train_pred)
    val_loss   = mean_squared_error(y_val_scaled, y_val_pred)
    train_losses.append(train_loss); val_losses.append(val_loss)

    print(f"Epoch {epoch+1:03}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

# Final Test Evaluation
y_train_pred = y_scaler.inverse_transform(model.predict(X_train_scaled))
y_test_pred  = y_scaler.inverse_transform(model.predict(X_test_scaled))

def evaluate_model(true, pred, name):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    
    print(f"{name}  MAE: {mae:.4f},  RMSE: {rmse:.4f},  R²: {r2:.4f}")

print("\nMODEL PERFORMANCE")
print("Training Performance:")
evaluate_model(y_train['CN'], y_train_pred[:,0], "CN Train")
evaluate_model(y_train['CM'], y_train_pred[:,1], "CM Train")

print("\nTesting Performance:")
evaluate_model(y_test['CN'], y_test_pred[:,0], "CN Test")
evaluate_model(y_test['CM'], y_test_pred[:,1], "CM Test")


# Loss Curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training & Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Scaled)")
plt.grid(True)
plt.legend()
plt.show()

# Actual vs Predicted Scatter Plots (CN & CM)
plt.figure(figsize=(6,5))
plt.scatter(y_test['CN'], y_test_pred[:,0], s=10, color='blue')
plt.plot(y_test['CN'], y_test['CN'], 'r')
plt.xlabel("Actual CN"); plt.ylabel("Predicted CN")
plt.title("MLP - CN Actual vs Predicted")
plt.grid(True); plt.show()

plt.figure(figsize=(6,5))
plt.scatter(y_test['CM'], y_test_pred[:,1], s=10, color='green')
plt.plot(y_test['CM'], y_test['CM'], 'orange')
plt.xlabel("Actual CM"); plt.ylabel("Predicted CM")
plt.title("MLP - CM Actual vs Predicted")
plt.grid(True); plt.show()




"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------- 1. Load data ---------------------------
file_path = "DRDL_aero_data_final.csv"
data = pd.read_csv(file_path)

# --------------------------- 2. Clean data ---------------------------
# Drop constant columns and the ID column
constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
data_cleaned = data.drop(columns=constant_columns + ['CASEID'])

# Force everything to numeric (NaNs will appear for non‑convertible entries)
data_numeric = data_cleaned.apply(pd.to_numeric, errors='coerce')

# --------------------------- 3. Feature / target selection ---------------------------
selected_features = ["ALPHA", "MACH", "X-C.P.", "nose length", "semi-span", "wing sweep"]
X = data_numeric[selected_features]                     # shape: (n_samples, 6)
y = data_numeric[['CN', 'CM']]                         # shape: (n_samples, 2)

# --------------------------- 4. Train / validation / test split ---------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42
)

# --------------------------- 5. Scaling ---------------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled   = x_scaler.transform(X_val)
X_test_scaled  = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled   = y_scaler.transform(y_val)
y_test_scaled  = y_scaler.transform(y_test)

# --------------------------- 6. Model definition ---------------------------
# One hidden layer with 64 neurons
model = MLPRegressor(
    hidden_layer_sizes=(64,),   # <-- single hidden layer
    activation='relu',
    solver='adam',
    max_iter=1,                 # we will manually loop over epochs
    warm_start=True,            # keep weights between .fit() calls
    random_state=42
)

# --------------------------- 7. Training loop ---------------------------
epochs = 500
train_losses = []
val_losses = []

print("\nTraining Started...\n")
for epoch in range(epochs):
    # One epoch = one call to partial_fit
    model.partial_fit(X_train_scaled, y_train_scaled)

    # Predictions for loss computation
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred   = model.predict(X_val_scaled)

    # MSE on the *scaled* targets (same metric used during training)
    train_loss = mean_squared_error(y_train_scaled, y_train_pred)
    val_loss   = mean_squared_error(y_val_scaled, y_val_pred)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:03}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

# --------------------------- 8. Final evaluation on original scale ---------------------------
y_train_pred = y_scaler.inverse_transform(model.predict(X_train_scaled))
y_test_pred  = y_scaler.inverse_transform(model.predict(X_test_scaled))

def evaluate_model(true, pred, name):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    print(f"{name}  MAE: {mae:.4f},  RMSE: {rmse:.4f},  R²: {r2:.4f}")

print("\nMODEL PERFORMANCE")
print("Training Performance:")
evaluate_model(y_train['CN'], y_train_pred[:, 0], "CN Train")
evaluate_model(y_train['CM'], y_train_pred[:, 1], "CM Train")

print("\nTesting Performance:")
evaluate_model(y_test['CN'], y_test_pred[:, 0], "CN Test")
evaluate_model(y_test['CM'], y_test_pred[:, 1], "CM Test")

# --------------------------- 9. Plot loss curves ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses,   label='Validation Loss')
plt.title("Training & Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Scaled)")
plt.grid(True)
plt.legend()
plt.show()

# --------------------------- 10. Actual vs. Predicted scatter plots ---------------------------
# CN
plt.figure(figsize=(6, 5))
plt.scatter(y_test['CN'], y_test_pred[:, 0], s=10, color='blue')
plt.plot(y_test['CN'], y_test['CN'], 'r')   # perfect‑fit line
plt.xlabel("Actual CN")
plt.ylabel("Predicted CN")
plt.title("MLP – CN Actual vs Predicted")
plt.grid(True)
plt.show()

# CM
plt.figure(figsize=(6, 5))
plt.scatter(y_test['CM'], y_test_pred[:, 1], s=10, color='green')
plt.plot(y_test['CM'], y_test['CM'], 'orange')
plt.xlabel("Actual CM")
plt.ylabel("Predicted CM")
plt.title("MLP – CM Actual vs Predicted")"""