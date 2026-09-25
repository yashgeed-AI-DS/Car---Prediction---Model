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
warnings.filterwarnings("ignore")

# ============================================================

# PAGE CONFIG

# ============================================================

st.set_page_config(
page_title="🚗 Car Price Predictor",
page_icon="🚗",
layout="wide"
)

# ============================================================

# TITLE

# ============================================================

st.title("🚗 Used Car Price Prediction")

st.markdown(
"**Linear Regression vs Ridge vs Lasso** — "
"Predict the price of a used car using Machine Learning."
)

st.markdown("---")

# ============================================================

# SIDEBAR

# ============================================================

st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
"Upload your CSV file",
type=["csv"],
help="Upload the car sales dataset used for training."
)

# ============================================================

# IF FILE IS NOT UPLOADED

# ============================================================

if uploaded_file is None:

    st.info(
    "👈 Please upload your car dataset from the sidebar."
)

st.markdown("""
### Expected Columns

| Column | Description |
|---|---|
| `Brand` | Car brand |
| `Price` | Car price |
| `Body` | Body type |
| `Mileage` | Distance driven |
| `EngineV` | Engine volume |
| `Engine Type` | Fuel type |
| `Registration` | Registration status |
| `Year` | Manufacturing year |
| `Model` | Car model |
""")

st.stop()


# ============================================================

# LOAD DATA

# ============================================================

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df_raw = load_data(uploaded_file)

# ============================================================

# CHECK REQUIRED COLUMNS

# ============================================================

required_columns = [
"Brand",
"Price",
"Body",
"Mileage",
"EngineV",
"Engine Type",
"Registration",
"Year",
"Model"
]

missing_columns = [
col for col in required_columns
if col not in df_raw.columns
]

if missing_columns:

    st.error(
    f"❌ Missing columns in dataset: "
    f"{', '.join(missing_columns)}"
)

st.stop()

# ============================================================

# CONVERT IMPORTANT NUMERIC COLUMNS

# ============================================================

numeric_columns = [
"Price",
"Mileage",
"EngineV",
"Year"
]

for column in numeric_columns:


    df_raw[column] = pd.to_numeric(
    df_raw[column],
    errors="coerce"
)


# ============================================================

# RAW DATA

# ============================================================

st.subheader("📊 Raw Data Preview")

st.dataframe(
df_raw.head(10),
use_container_width=True
)

st.caption(
f"Dataset shape: {df_raw.shape[0]} rows × "
f"{df_raw.shape[1]} columns"
)

# ============================================================

# PREPROCESSING

# ============================================================

st.markdown("---")

st.subheader("🔧 Data Preprocessing")

df = df_raw.copy()

with st.expander(
"View Step-by-Step Preprocessing",
expanded=False
):


# --------------------------------------------------------
# STEP 1: REMOVE MISSING PRICE / ENGINEV
# --------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=["Price", "EngineV"]
)

removed = before - len(df)

st.write(
    f"✅ **Step 1:** Removed {removed} rows "
    "with missing Price or EngineV."
)


# --------------------------------------------------------
# STEP 2: REMOVE INVALID PRICE
# --------------------------------------------------------

before = len(df)

df = df[df["Price"] > 0]

removed = before - len(df)

st.write(
    f"✅ **Step 2:** Removed {removed} rows "
    "where Price was zero or negative."
)


# --------------------------------------------------------
# STEP 3: REMOVE ENGINEV OUTLIERS
# --------------------------------------------------------

before = len(df)

df = df[df["EngineV"] <= 10]

removed = before - len(df)

st.write(
    f"✅ **Step 3:** Removed {removed} rows "
    "where EngineV > 10."
)


# --------------------------------------------------------
# STEP 4: LOG TRANSFORMATION
# --------------------------------------------------------

df["Log_price"] = np.log(
    df["Price"]
)

st.write(
    "✅ **Step 4:** Applied log transformation "
    "`Log_price = log(Price)`."
)


# --------------------------------------------------------
# CATEGORICAL COLUMNS
# --------------------------------------------------------

categorical_columns = [
    "Brand",
    "Body",
    "Engine Type",
    "Registration"
]


# --------------------------------------------------------
# SAVE ORIGINAL CATEGORIES
# --------------------------------------------------------

category_options = {}

for column in categorical_columns:

    category_options[column] = sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# --------------------------------------------------------
# STEP 5: DROP MODEL AND PRICE
# --------------------------------------------------------

df = df.drop(
    columns=["Model", "Price"]
)

st.write(
    "✅ **Step 5:** Dropped `Model` and "
    "original `Price`."
)


# --------------------------------------------------------
# STEP 6: LABEL ENCODING
# --------------------------------------------------------

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].fillna("Unknown").astype(str)
    )

    encoders[column] = encoder


st.write(
    "✅ **Step 6:** Label encoded categorical columns:"
)

st.code(
    ", ".join(categorical_columns)
)
```

st.success(
f"Dataset ready: {df.shape[0]} rows × "
f"{df.shape[1]} columns"
)

# ============================================================

# EDA

# ============================================================

st.markdown("---")

st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

# ============================================================

# PRICE DISTRIBUTION

# ============================================================

with col1:

```
st.markdown("**Price Distribution**")

fig1, ax1 = plt.subplots()

ax1.hist(
    df_raw["Price"].dropna(),
    bins=50,
    edgecolor="white"
)

ax1.set_xlabel("Price")
ax1.set_ylabel("Count")
ax1.set_title("Original Price Distribution")

st.pyplot(fig1)

plt.close(fig1)
```

# ============================================================

# LOG PRICE DISTRIBUTION

# ============================================================

with col2:

```
st.markdown("**Log Price Distribution**")

fig2, ax2 = plt.subplots()

ax2.hist(
    df["Log_price"],
    bins=50,
    edgecolor="white"
)

ax2.set_xlabel("Log Price")
ax2.set_ylabel("Count")
ax2.set_title("Log Price Distribution")

st.pyplot(fig2)

plt.close(fig2)


# ============================================================

# CORRELATION HEATMAP

# ============================================================

st.markdown("**Correlation Heatmap**")

fig3, ax3 = plt.subplots(
figsize=(10, 5)
)

sns.heatmap(
df.corr(numeric_only=True),
annot=True,
fmt=".2f",
cmap="coolwarm",
ax=ax3
)

st.pyplot(fig3)

plt.close(fig3)

# ============================================================

# MODEL PARAMETERS

# ============================================================

st.markdown("---")

st.subheader("🤖 Model Training & Comparison")

st.sidebar.header("⚙️ Model Parameters")

test_size = st.sidebar.slider(
"Test Set Size (%)",
min_value=10,
max_value=40,
value=20,
step=5
) / 100

random_state = st.sidebar.number_input(
"Random State",
value=42,
min_value=0
)

ridge_alpha = st.sidebar.slider(
"Ridge α",
0.01,
100.0,
value=1.0,
step=0.1
)

lasso_alpha = st.sidebar.slider(
"Lasso α",
0.0001,
1.0,
value=0.001,
step=0.0001,
format="%.4f"
)

# ============================================================

# TRAINING DATA

# ============================================================

X = df.drop(
["Log_price"],
axis=1
)

y = df["Log_price"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=test_size,
random_state=int(random_state)
)

# ============================================================

# TRAIN MODELS

# ============================================================

linear_reg = LinearRegression()

ridge_reg = Ridge(
alpha=ridge_alpha
)

lasso_reg = Lasso(
alpha=lasso_alpha
)

linear_reg.fit(
X_train,
y_train
)

ridge_reg.fit(
X_train,
y_train
)

lasso_reg.fit(
X_train,
y_train
)

# ============================================================

# PREDICTIONS

# ============================================================

y_pred_linear = linear_reg.predict(
X_test
)

y_pred_ridge = ridge_reg.predict(
X_test
)

y_pred_lasso = lasso_reg.predict(
X_test
)

# ============================================================

# METRICS

# ============================================================

def get_metrics(y_true, y_pred):

```
return {
    "R² Score": round(
        r2_score(
            y_true,
            y_pred
        ),
        4
    ),

    "RMSE": round(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),
        4
    ),

    "MAE": round(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        ),
        4
    )
}

metrics = {


"Linear Regression": get_metrics(
    y_test,
    y_pred_linear
),

"Ridge Regression": get_metrics(
    y_test,
    y_pred_ridge
),

"Lasso Regression": get_metrics(
    y_test,
    y_pred_lasso
)


}

metrics_df = (
pd.DataFrame(metrics)
.T
.reset_index()
.rename(
columns={
"index": "Model"
}
)
)

# ============================================================

# MODEL COMPARISON

# ============================================================

st.markdown("### 📋 Model Comparison")

st.dataframe(
metrics_df,
use_container_width=True
)

best_model = metrics_df.loc[
metrics_df["R² Score"].idxmax(),
"Model"
]

st.success(
f"🏆 Highest R² Score: **{best_model}**"
)

# ============================================================

# ACTUAL VS PREDICTED

# ============================================================

st.markdown("---")

st.subheader("📉 Actual vs Predicted")

fig, axes = plt.subplots(
1,
3,
figsize=(18, 5)
)

models_plot = [

```
(
    "Linear Regression",
    y_pred_linear
),

(
    "Ridge Regression",
    y_pred_ridge
),

(
    "Lasso Regression",
    y_pred_lasso
)

]

for ax, (name, y_pred) in zip(
axes,
models_plot
):


ax.scatter(
    y_test,
    y_pred,
    alpha=0.4,
    s=15
)

ax.plot(
    [
        y_test.min(),
        y_test.max()
    ],
    [
        y_test.min(),
        y_test.max()
    ],
    "r--",
    lw=2
)

ax.set_xlabel(
    "Actual Log Price"
)

ax.set_ylabel(
    "Predicted Log Price"
)

ax.set_title(name)

ax.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================

# INTERACTIVE CAR PRICE PREDICTION

# ============================================================

st.markdown("---")

st.subheader(
"🚗 Predict Price of Your Car"
)

st.markdown(
"Enter the details of the car using "
"the interactive controls below."
)

# ============================================================

# PREDICTION FORM

# ============================================================

with st.form(
"car_prediction_form"
):


st.markdown(
    "### 🔧 Car Details"
)


# --------------------------------------------------------
# ROW 1
# --------------------------------------------------------

col1, col2, col3 = st.columns(3)


# BRAND

with col1:

    selected_brand = st.selectbox(
        "🏷️ Brand",
        category_options["Brand"]
    )


# BODY

with col2:

    selected_body = st.selectbox(
        "🚘 Body Type",
        category_options["Body"]
    )


# ENGINE TYPE

with col3:

    selected_engine_type = st.selectbox(
        "⛽ Engine Type",
        category_options["Engine Type"]
    )


# --------------------------------------------------------
# ROW 2
# --------------------------------------------------------

col1, col2, col3 = st.columns(3)


# REGISTRATION

with col1:

    selected_registration = st.selectbox(
        "📄 Registration",
        category_options["Registration"]
    )


# MILEAGE

with col2:

    mileage_min = int(
        max(
            0,
            df_raw["Mileage"].min()
        )
    )

    mileage_max = int(
        df_raw["Mileage"].max()
    )

    mileage_default = int(
        df_raw["Mileage"].median()
    )

    mileage = st.slider(
        "🛣️ Mileage",
        min_value=mileage_min,
        max_value=mileage_max,
        value=mileage_default
    )


# ENGINE VOLUME

with col3:

    engine_min = float(
        max(
            0.1,
            df_raw["EngineV"].min()
        )
    )

    engine_max = float(
        min(
            10.0,
            df_raw["EngineV"].max()
        )
    )

    engine_default = float(
        df_raw["EngineV"].median()
    )

    engine_default = min(
        max(
            engine_default,
            engine_min
        ),
        engine_max
    )

    engine_volume = st.slider(
        "🔧 Engine Volume (L)",
        min_value=engine_min,
        max_value=engine_max,
        value=engine_default,
        step=0.1
    )


# --------------------------------------------------------
# YEAR
# --------------------------------------------------------

year_min = int(
    df_raw["Year"].min()
)

year_max = int(
    df_raw["Year"].max()
)

year_default = int(
    df_raw["Year"].median()
)

year = st.slider(
    "📅 Manufacturing Year",
    min_value=year_min,
    max_value=year_max,
    value=year_default
)


# --------------------------------------------------------
# MODEL CHOICE
# --------------------------------------------------------

model_choice = st.selectbox(
    "🤖 Select Prediction Model",
    [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ]
)


st.markdown("")


# --------------------------------------------------------
# SUBMIT BUTTON
# --------------------------------------------------------

submitted = st.form_submit_button(
    "🚀 Predict Car Price",
    use_container_width=True
)


# ============================================================

# PREDICTION

# ============================================================

if submitted:


try:

    # ----------------------------------------------------
    # ENCODE CATEGORICAL VALUES
    # ----------------------------------------------------

    brand_encoded = encoders[
        "Brand"
    ].transform(
        [str(selected_brand)]
    )[0]


    body_encoded = encoders[
        "Body"
    ].transform(
        [str(selected_body)]
    )[0]


    engine_type_encoded = encoders[
        "Engine Type"
    ].transform(
        [str(selected_engine_type)]
    )[0]


    registration_encoded = encoders[
        "Registration"
    ].transform(
        [str(selected_registration)]
    )[0]


    # ----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # ----------------------------------------------------

    input_data = {

        "Brand": brand_encoded,

        "Body": body_encoded,

        "Mileage": mileage,

        "EngineV": engine_volume,

        "Engine Type": engine_type_encoded,

        "Registration": registration_encoded,

        "Year": year
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    # Ensure same feature order
    # as training data

    input_df = input_df[
        X.columns
    ]


    # ----------------------------------------------------
    # SELECT MODEL
    # ----------------------------------------------------

    model_map = {

        "Linear Regression":
            linear_reg,

        "Ridge Regression":
            ridge_reg,

        "Lasso Regression":
            lasso_reg
    }


    selected_model = model_map[
        model_choice
    ]


    # ----------------------------------------------------
    # PREDICT LOG PRICE
    # ----------------------------------------------------

    log_prediction = selected_model.predict(
        input_df
    )[0]


    # ----------------------------------------------------
    # CONVERT LOG PRICE TO ORIGINAL PRICE
    # ----------------------------------------------------

    predicted_price = np.exp(
        log_prediction
    )


    # ----------------------------------------------------
    # RESULT
    # ----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🎯 Prediction Result"
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "💰 Estimated Car Price",
            f"${predicted_price:,.2f}"
        )


    with result_col2:

        st.metric(
            "🤖 Model Used",
            model_choice
        )


    st.success(
        "✅ Prediction completed successfully!"
    )


    st.info(
        f"Log Price Prediction: "
        f"{log_prediction:.4f}"
    )


    # ----------------------------------------------------
    # SELECTED CAR DETAILS
    # ----------------------------------------------------

    st.markdown(
        "### 🚘 Selected Car Details"
    )


    details_df = pd.DataFrame({

        "Feature": [

            "Brand",
            "Body Type",
            "Mileage",
            "Engine Volume",
            "Engine Type",
            "Registration",
            "Year"
        ],

        "Value": [

            selected_brand,

            selected_body,

            f"{mileage:,} km",

            f"{engine_volume:.1f} L",

            selected_engine_type,

            selected_registration,

            year
        ]
    })


    st.dataframe(
        details_df,
        hide_index=True,
        use_container_width=True
    )


except Exception as e:

    st.error(
        "❌ Prediction failed."
    )

    st.code(
        str(e)
    )


# ============================================================

# FOOTER

# ============================================================

st.markdown("---")

st.caption(
"Built with ❤️ using Streamlit · "
"Linear Regression · Ridge Regression · "
"Lasso Regression"
)
