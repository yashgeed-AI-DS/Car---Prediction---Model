import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Price Prediction")
st.markdown("**Linear Regression vs Ridge vs Lasso** — Compare all three models on real car sales data")
st.markdown("---")

# ─── Upload CSV ─────────────────────────────────────────────────────────────
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file", type=["csv"],
    help="Expected: '1.04. Real-life example.csv' or similar car sales dataset"
)

if uploaded_file is None:
    st.info("👈 Please upload a CSV file from the sidebar to get started.")
    st.markdown("""
    ### Expected Columns in the Dataset:
    | Column | Description |
    |--------|-------------|
    | `Brand` | Car brand (e.g., BMW, Audi) |
    | `Price` | Car price (target variable) |
    | `Body` | Body type (e.g., sedan, SUV) |
    | `Mileage` | Distance driven |
    | `EngineV` | Engine volume in litres |
    | `Engine Type` | Fuel type (Petrol/Diesel etc.) |
    | `Registration` | Registered or not |
    | `Year` | Year of manufacture |
    | `Model` | Car model name |
    """)
    st.stop()

# ─── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    return df

df_raw = load_and_preprocess(uploaded_file)

st.subheader("📊 Raw Data Preview")
st.dataframe(df_raw.head(10), use_container_width=True)
st.caption(f"Dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ─── Preprocessing ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔧 Data Preprocessing Steps")

with st.expander("View Step-by-Step Preprocessing", expanded=True):
    df = df_raw.copy()

    # Step 1 — Drop missing values
    before = len(df)
    df = df.dropna(subset=['Price', 'EngineV'])
    after = len(df)
    st.write(f"✅ **Step 1 — Dropped missing values:** Removed {before - after} rows (Price/EngineV was NaN)")

    # Step 2 — Remove outliers
    before2 = len(df)
    df = df[df['EngineV'] <= 10]
    st.write(f"✅ **Step 2 — Removed outliers:** Dropped {before2 - len(df)} rows where EngineV > 10")

    # Step 3 — Log transform Price
    df['Log_price'] = np.log(df['Price'])
    st.write("✅ **Step 3 — Log transformation:** Applied `np.log(Price)` → `Log_price` (normalises skewed distribution)")

    # Step 4 — Drop Model & Price
    df = df.drop(columns=['Model', 'Price'])
    st.write("✅ **Step 4 — Dropped columns:** Removed `Model` (too many unique values) and original `Price`")

    # Step 5 — Encode categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    st.write(f"✅ **Step 5 — Label Encoding:** Encoded columns: `{'`, `'.join(cat_cols)}`")

    st.success(f"Final dataset ready: {df.shape[0]} rows × {df.shape[1]} columns")

# ─── EDA ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Price Distribution (Original)**")
    fig1, ax1 = plt.subplots()
    ax1.hist(df_raw['Price'].dropna(), bins=50, color='steelblue', edgecolor='white')
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Count")
    ax1.set_title("Skewed Distribution")
    st.pyplot(fig1)

with col2:
    st.markdown("**Log Price Distribution (After Transformation)**")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Log_price'], bins=50, color='seagreen', edgecolor='white')
    ax2.set_xlabel("Log Price")
    ax2.set_ylabel("Count")
    ax2.set_title("Normal-like Distribution ✅")
    st.pyplot(fig2)

st.markdown("**Correlation Heatmap**")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# ─── Model Training ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Model Training & Comparison")

# Sidebar — model params
st.sidebar.header("⚙️ Model Parameters")
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
random_state = st.sidebar.number_input("Random State", value=42, min_value=0)
ridge_alpha = st.sidebar.slider("Ridge α (regularization)", 0.01, 100.0, value=1.0, step=0.1)
lasso_alpha = st.sidebar.slider("Lasso α (regularization)", 0.0001, 1.0, value=0.001, step=0.0001, format="%.4f")

X = df.drop(['Log_price'], axis=1)
y = df['Log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

# Train models
linear_reg = LinearRegression().fit(X_train, y_train)
ridge_reg = Ridge(alpha=ridge_alpha).fit(X_train, y_train)
lasso_reg = Lasso(alpha=lasso_alpha).fit(X_train, y_train)

# Predictions
y_pred_linear = linear_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)

# Metrics
def get_metrics(y_true, y_pred):
    return {
        "R² Score": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(np.mean(np.abs(y_true - y_pred)), 4)
    }

metrics = {
    "Linear Regression": get_metrics(y_test, y_pred_linear),
    "Ridge Regression": get_metrics(y_test, y_pred_ridge),
    "Lasso Regression": get_metrics(y_test, y_pred_lasso),
}

metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

st.markdown("### 📋 Model Comparison Table")
st.dataframe(metrics_df, use_container_width=True)

# Best model highlight
best_model = metrics_df.loc[metrics_df["R² Score"].idxmax(), "Model"]
st.success(f"🏆 **Best Model:** {best_model} (highest R² Score)")

# ─── Actual vs Predicted Plots ───────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Actual vs Predicted Plots")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_plot = [
    ("Linear Regression", y_pred_linear, "steelblue"),
    ("Ridge Regression",  y_pred_ridge,  "darkorange"),
    ("Lasso Regression",  y_pred_lasso,  "seagreen"),
]

for ax, (name, y_pred, color) in zip(axes, models_plot):
    ax.scatter(y_test, y_pred, alpha=0.4, color=color, s=15)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Log Price")
    ax.set_ylabel("Predicted Log Price")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ─── Predict on New Input ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Predict Price for a New Car")

feature_cols = X.columns.tolist()
with st.form("prediction_form"):
    st.markdown("Enter car details below:")
    input_vals = {}
    cols = st.columns(3)
    for i, col_name in enumerate(feature_cols):
        with cols[i % 3]:
            input_vals[col_name] = st.number_input(
                col_name,
                value=float(X[col_name].median()),
                format="%.2f"
            )

    model_choice = st.selectbox("Choose model for prediction", ["Linear Regression", "Ridge Regression", "Lasso Regression"])
    submitted = st.form_submit_button("Predict Price 🚀")

if submitted:
    input_df = pd.DataFrame([input_vals])
    model_map = {
        "Linear Regression": linear_reg,
        "Ridge Regression":  ridge_reg,
        "Lasso Regression":  lasso_reg
    }
    log_pred = model_map[model_choice].predict(input_df)[0]
    actual_price = np.exp(log_pred)
    st.metric(label=f"Predicted Car Price ({model_choice})", value=f"${actual_price:,.2f}")
    st.caption(f"Log price predicted: {log_pred:.4f} → Converted back: e^{log_pred:.4f} = ${actual_price:,.2f}")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Linear, Ridge & Lasso Regression · WezInsights")
