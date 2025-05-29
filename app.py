import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, norm

st.set_page_config(page_title="Motion Data Dashboard", layout="wide")
st.title("📊 Motion Data Dashboard")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File successfully uploaded!")

    with st.expander("📌 Data Info"):
        st.write(f"📏 Data Shape: {df.shape}")
        st.write("📊 Column Types:")
        st.write(df.dtypes)

    with st.expander("💔 Missing Value Analysis"):
        st.write(df.isnull().sum())

    with st.expander("🔍 Numerical Columns"):
        num_cols = df.select_dtypes(include='number').columns
        st.write(num_cols)

    with st.expander("🧰 Filter Options"):
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        st.subheader("Correlation Matrix:")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

    with st.expander("🧪 Select Column & Plot"):
        selected = st.selectbox("Choose a numeric column:", num_cols)
        plot_type = st.radio("Plot type:", ["Histogram", "Boxplot", "Density Plot"])

        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            sns.histplot(df[selected], kde=False, ax=ax)
        elif plot_type == "Boxplot":
            sns.boxplot(x=df[selected], ax=ax)
        else:
            sns.kdeplot(df[selected], fill=True, ax=ax)
        st.pyplot(fig)

    with st.expander("🌱 Sampling & Central Limit Theorem"):
        st.write("Demonstration of the Central Limit Theorem via sampling.")
        sample_col = st.selectbox("Choose a numeric column for sampling:", num_cols)
        sample_size = st.slider("Sample size (n):", 10, 500, 100)
        num_samples = st.slider("Number of samples:", 10, 200, 50)

        sample_means = [df[sample_col].dropna().sample(sample_size).mean() for _ in range(num_samples)]

        fig, ax = plt.subplots()
        sns.histplot(sample_means, kde=True, ax=ax)
        ax.set_title(f"Sampling Distribution of Sample Mean ({sample_col})")
        st.pyplot(fig)

    with st.expander("📉 Maximum Likelihood Estimation (MLE)"):
        mle_col = st.selectbox("Select a numeric column for MLE:", num_cols)
        data = df[mle_col].dropna()
        mu, std = norm.fit(data)

        fig, ax = plt.subplots()
        sns.histplot(data, kde=False, stat="density", ax=ax, label="Empirical")
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'r', linewidth=2, label=f'N(μ={mu:.2f}, σ={std:.2f})')
        ax.set_title("MLE Fit to Data")
        ax.legend()
        st.pyplot(fig)

        st.write(f"**Estimated μ (mean):** {mu:.4f}")
        st.write(f"**Estimated σ (std dev):** {std:.4f}")

    with st.expander("🔬 Hypothesis Testing (t-test)"):
        st.write("This section performs a t-test between two groups for a selected numerical column.")

        cat_cols = df.select_dtypes(include='object').columns
        cat_summary = {col: df[col].nunique() for col in cat_cols}
        binary_cat_cols = [col for col, count in cat_summary.items() if count == 2]

        st.write("🧾 Categorical columns and their unique value counts:")
        st.json(cat_summary)

        if binary_cat_cols:
            cat_col = st.selectbox("Select a categorical column (must have 2 unique values):", binary_cat_cols)
            num_col = st.selectbox("Select a numeric column:", num_cols)

            group_values = df[cat_col].unique()
            group1 = df[df[cat_col] == group_values[0]][num_col]
            group2 = df[df[cat_col] == group_values[1]][num_col]

            if group1.empty or group2.empty:
                st.warning("⚠️ One of the groups is empty. Please check your column selection.")
            else:
                stat, pval = ttest_ind(group1, group2, equal_var=False)
                st.write(f"**T-statistic:** {stat:.4f}")
                st.write(f"**P-value:** {pval:.4f}")

                if pval < 0.05:
                    st.success("✅ Significant difference found (reject null hypothesis).")
                else:
                    st.info("🟡 No significant difference found (fail to reject null hypothesis).")
        else:
            st.warning("⚠️ No categorical column with exactly two unique values found.")

    with st.expander("📈 Simple Linear Regression"):
        x_var = st.selectbox("Select independent variable (X):", num_cols)
        y_var = st.selectbox("Select dependent variable (Y):", num_cols)

        X = df[x_var].dropna()
        Y = df[y_var].dropna()

        if not X.empty and not Y.empty and len(X) == len(Y):
            coeffs = np.polyfit(X, Y, deg=1)
            a, b = coeffs
            fig, ax = plt.subplots()
            ax.scatter(X, Y, label="Data points")
            ax.plot(X, a * X + b, color="red", label=f"y = {a:.2f}x + {b:.2f}")
            ax.set_title("Linear Regression")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("⚠️ Ensure selected columns are valid and aligned in length.")
else:
    st.info("⬅️ Please upload a CSV file from the sidebar.")
