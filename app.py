import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Motion Data Dashboard", layout="wide")
st.title("📊 Motion Data Dashboard")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File successfully uploaded!")

    # Data Info
    with st.expander("📌 Data Info"):
        st.write(f"🔢 Data Shape: {df.shape}")
        st.write("📚 Column Types:")
        st.write(df.dtypes)

    # Missing Values
    with st.expander("💔 Missing Value Analysis"):
        st.write(df.isnull().sum())

    # Numerical Columns
    with st.expander("🔍 Numerical Columns"):
        st.write(df.select_dtypes(include='number').columns)

    # Descriptive Stats + Correlation
    with st.expander("📊 Filter Options"):
        st.write("📈 Descriptive Statistics")
        st.write(df.describe())

        st.write("🔗 Correlation Matrix:")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Plotting
    with st.expander("📎 Select Column & Plot"):
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

    # Sampling + CLT
    with st.expander("🧪 Sampling & Central Limit Theorem"):
        selected_col = st.selectbox("Choose a numeric column for sampling:", num_cols, key="sampling")
        sample_size = st.slider("Sample size (n):", 10, 500, 100, 10)
        num_samples = st.slider("Number of samples:", 10, 200, 50, 10)

        sample_means = []
        for _ in range(num_samples):
            sample = df[selected_col].dropna().sample(sample_size, replace=True)
            sample_means.append(sample.mean())

        fig, ax = plt.subplots()
        sns.histplot(sample_means, kde=True, ax=ax)
        ax.set_title(f"Sampling Distribution of Sample Mean ({selected_col})")
        st.pyplot(fig)

    # MLE
    with st.expander("📐 Maximum Likelihood Estimation (MLE)"):
        selected_mle_col = st.selectbox("Select a numeric column for MLE:", num_cols, key="mle")
        data = df[selected_mle_col].dropna()
        mu_mle = np.mean(data)
        sigma_mle = np.std(data)

        x_vals = np.linspace(data.min(), data.max(), 100)
        normal_dist = (1 / (sigma_mle * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu_mle) / sigma_mle)**2)

        fig, ax = plt.subplots()
        sns.histplot(data, stat="density", bins=30, kde=False, ax=ax, label="Empirical")
        ax.plot(x_vals, normal_dist, color='red', label=f"N(μ={mu_mle:.2f}, σ={sigma_mle:.2f})")
        ax.set_title("MLE Fit to Data")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"**Estimated μ (mean):** {mu_mle:.4f}  \n**Estimated σ (std dev):** {sigma_mle:.4f}")

    # Hypothesis Testing
    with st.expander("🧪 Hypothesis Testing (t-test)"):
        cat_cols = df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            group_col = st.selectbox("Choose a categorical column:", cat_cols, key="ht_group")
            test_col = st.selectbox("Choose a numeric column for comparison:", num_cols, key="ht_test")
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                group1 = df[df[group_col] == groups[0]][test_col].dropna()
                group2 = df[df[group_col] == groups[1]][test_col].dropna()
                t_stat, p_val = stats.ttest_ind(group1, group2)
                st.write(f"Groups: {groups[0]} vs. {groups[1]}")
                st.write(f"**t-statistic:** {t_stat:.4f}")
                st.write(f"**p-value:** {p_val:.4f}")
                if p_val < 0.05:
                    st.success("📢 Result: Statistically significant difference.")
                else:
                    st.info("ℹ️ Result: No statistically significant difference.")
            else:
                st.warning("Please choose a categorical column with exactly two groups.")

    # Simple Linear Regression
    with st.expander("📈 Simple Linear Regression"):
        x_col = st.selectbox("Select independent variable (X):", num_cols, key="reg_x")
        y_col = st.selectbox("Select dependent variable (Y):", num_cols, key="reg_y")
        x = df[x_col].dropna()
        y = df[y_col].dropna()
        if len(x) == len(y) and len(x) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            fig, ax = plt.subplots()
            ax.scatter(x, y, label='Data points')
            ax.plot(x, intercept + slope * x, 'r', label=f'y = {intercept:.2f} + {slope:.2f}x')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Linear Regression")
            ax.legend()
            st.pyplot(fig)
            st.markdown(f"**R-squared:** {r_value**2:.4f}  \n**p-value:** {p_value:.4f}")

else:
    st.info("👉 Please upload a CSV file from the left panel.")
