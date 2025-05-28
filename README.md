# Motion Data Dashboard

This project presents an interactive data dashboard developed using Streamlit for visualizing and analyzing motion capture data. The dashboard provides basic data exploration, visualization, and filtering functionalities suitable for initial exploratory data analysis (EDA).

## Features

- Upload and preview motion capture datasets in CSV format
- Display dataset shape and data types
- Analyze missing values
- List and inspect numerical columns
- Generate descriptive statistics for selected columns
- Visualize correlations using a heatmap
- Create customizable plots (Histogram, Boxplot, Density Plot)
- Filter the dataset based on user-selected columns and values

## Technologies Used

- Python 3.10
- Streamlit
- Pandas
- Matplotlib
- Seaborn

## How to Run

1. Clone the repository:

2. Install the required libraries:

3. Run the Streamlit app:

## File Structure

- `app.py` – Main Streamlit application
- `requirements.txt` – Required Python packages

## Notes

- The dashboard is optimized for wide layout screens.
- It accepts only CSV files with a maximum size of 200MB.
- The dataset must contain numeric columns to enable plotting and statistical analysis.

## License

This project is intended for academic and educational purposes.
