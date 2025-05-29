import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, ttest_ind
import numpy as np

st.set_page_config(page_title="Motion Data Dashboard", layout="wide")
st.title("📊 Motion Data Dashboard")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File successfully uploaded!")

    with st.expander("📌 Data Info"):
        st.write(f"📏 Data Shape: {df.shape}")
        st.write("🧱 Column Types:")
        st.write(df.dtypes)

    with st.expander("💔 Missing Value Analysis"):
        st.write(df.isnull().sum())

    with st.expander("🔍 Numerical Columns"):
        st.write(df.select_dtypes(include='number').columns)

    with st.expander("📊 Filter Options"):
        st.subheader("Descriptive Statistics:")
        st.write(df.describe())

        st.subheader("📌 Correlation Matrix:")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with st.expander("🧮 Select Column & Plot"):
        num_cols = df.select_dtypes(include='number').columns
        selected = st.selectbox("Choose a numeric column:", num_cols)
        plot_type = st.radio("Plot type:", ["Histogram", "Boxplot", "Density Plot"])

        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            sns.histplot(df[selected], kde=True, ax=ax)
        elif plot_type == "Boxplot":
            sns.boxplot(x=df[selected], ax=ax)
        else:
            sns.kdeplot(df[selected], fill=True, ax=ax)
        st.pyplot(fig)

    with st.expander("🧪 Sampling & Central Limit Theorem"):
        sample_col = st.selectbox("Choose a numeric column for sampling:", num_cols)
        n = st.slider("Sample size (n):", 10, 500, 100)
        k = st.slider("Number of samples:", 10, 200, 50)

        sample_means = [df[sample_col].dropna().sample(n).mean() for _ in range(k)]
        fig, ax = plt.subplots()
        sns.histplot(sample_means, kde=True, ax=ax)
        ax.set_title(f"Sampling Distribution of Sample Mean ({sample_col})")
        st.pyplot(fig)

    with st.expander("📈 Maximum Likelihood Estimation (MLE)"):
        mle_col = st.selectbox("Select a numeric column for MLE:", num_cols)
        data = df[mle_col].dropna()
        mu, std = data.mean(), data.std()

        fig, ax = plt.subplots()
        sns.histplot(data, stat="density", bins=30, color="steelblue", label="Empirical", ax=ax)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, "r", label=f"N(μ={mu:.2f}, σ={std:.2f})")
        ax.set_title("MLE Fit to Data")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"**Estimated μ (mean):** {mu:.4f}")
        st.markdown(f"**Estimated σ (std dev):** {std:.4f}")

    with st.expander("🔬 Hypothesis Testing (t-test)"):
        cat_cols = df.select_dtypes(include='object').columns
        binary_cat_cols = [col for col in cat_cols if df[col].nunique() == 2]

        if binary_cat_cols:
            cat_col = st.selectbox("Choose a categorical column:", binary_cat_cols)
            ttest_col = st.selectbox("Choose a numeric column for comparison:", num_cols)

            group1 = df[df[cat_col] == df[cat_col].unique()[0]][ttest_col]
            group2 = df[df[cat_col] == df[cat_col].unique()[1]][ttest_col]

            stat, pval = ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)
            st.write(f"T-statistic: {stat:.4f}")
            st.write(f"P-value: {pval:.4f}")

            if pval < 0.05:
                st.success("✅ Statistically significant difference (reject null hypothesis).")
            else:
                st.info("🟡 No significant difference (fail to reject null hypothesis).")
        else:
            st.warning("⚠️ No suitable categorical column with exactly two unique values.")

    with st.expander("📉 Simple Linear Regression"):
        x_var = st.selectbox("Select independent variable (X):", num_cols)
        y_var = st.selectbox("Select dependent variable (Y):", num_cols, index=1)

        fig, ax = plt.subplots()
        sns.regplot(x=df[x_var], y=df[y_var], ax=ax, line_kws={"color": "red"})
        ax.set_title("Linear Regression")
        st.pyplot(fig)

else:
    st.info("📎 Please upload a CSV file from above.")
