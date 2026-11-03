import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

##1.Load Dataset
FILE_NAME = "File.csv"
print("\nLoading dataset...")
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_FOLDER, FILE_NAME)

if not os.path.isfile(CSV_PATH):
    print("ERROR: CSV file not found!")
    print("Looking for:", CSV_PATH)
    print("Files in folder:", os.listdir(SCRIPT_FOLDER))
    sys.exit(1)

try:
    df = pd.read_csv(CSV_PATH)
except Exception as error:
    print("ERROR loading CSV:", error)
    sys.exit(1)

df.columns = df.columns.str.strip()

if df.empty or len(df.columns) == 0:
    print("ERROR: The CSV contains no usable data.")
    sys.exit(1)

print("Dataset loaded successfully!")
print("CSV location:", CSV_PATH)
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

##2.Identify Country and Year Columns
entity_col = None
year_col = None

for col in df.columns:
    name = col.strip().lower()

    if name in ["entity", "country", "location"]:
        entity_col = col

    if name == "year":
        year_col = col

if entity_col is None or year_col is None:
    print("\nERROR: Country/entity or year column not found.")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

##3.Identify Mental Health Disorder Columns
disorders = {}

keywords = {
    "Schizophrenia": "schizophrenia",
    "Depression": "depress",
    "Anxiety": "anxiety",
    "Bipolar Disorder": "bipolar",
    "Eating Disorder": "eating"
}

for disorder, keyword in keywords.items():
    for col in df.columns:
        if keyword in col.lower():
            disorders[disorder] = col
            break

if not disorders:
    print("ERROR: No mental health columns were detected.")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

for col in disorders.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

df = df.dropna(subset=[entity_col, year_col]).copy()

print("\nDisorders detected:")
for disorder in disorders:
    print("-", disorder)

print("Records available for analysis:", len(df))

##4.Dataset Summary
def show_summary():
    print("\n========== DATASET SUMMARY ==========")

    print("\nDataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df[list(disorders.values())].describe().round(4))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print(
        "Duplicate entity-year combinations:",
        df.duplicated(subset=[entity_col, year_col]).sum()
    )

    print("\nUnique entities:", df[entity_col].nunique())
    print("Earliest year:", int(df[year_col].min()))
    print("Latest year:", int(df[year_col].max()))

##5. Data Cleaning
def clean_data():
    global df

    print("\n========== DATA CLEANING ==========")

    print("Rows before cleaning:", len(df))

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    duplicates = df.duplicated().sum()
    print("\nDuplicate rows:", duplicates)

    df = df.drop_duplicates().copy()

    df = df.sort_values(
        by=[entity_col, year_col]
    ).reset_index(drop=True)

    print("\nRows after cleaning:", len(df))
    print("Rows removed:", duplicates)

    print("\nNegative prevalence values:")
    for name, col in disorders.items():
        print(name + ":", int((df[col] < 0).sum()))

    print("\nData cleaning completed.")
    print("Missing country codes are retained.")
    print("Missing prevalence values are not automatically replaced.")

##6.Choose Disorder
def choose_disorder():
    names = list(disorders.keys())

    print("\nChoose a mental health disorder:")

    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")

    try:
        choice = int(input("Enter choice: "))

        if 1 <= choice <= len(names):
            return names[choice - 1]

    except ValueError:
        pass

    print("Invalid choice.")
    return None

#7.Average Prevalence Bar Graph
def average_disorder_graph():
    averages = df[list(disorders.values())].mean()
    averages.index = list(disorders.keys())

    print("\nAverage prevalence:")
    print(averages.round(4))

    averages.plot(kind="bar", figsize=(9, 5))

    plt.title("Average Prevalence of Mental Health Disorders")
    plt.xlabel("Mental Health Disorder")
    plt.ylabel("Average prevalence (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

##8.Horizontal Bar Graph
def horizontal_bar_graph():
    averages = df[list(disorders.values())].mean()
    averages.index = list(disorders.keys())

    averages.sort_values().plot(
        kind="barh", figsize=(8, 5)
    )

    plt.title("Comparison of Mental Health Disorders")
    plt.xlabel("Average prevalence (%)")
    plt.ylabel("Disorder")
    plt.tight_layout()
    plt.show()

##9.Yearly Trend of One Disorder
def trend_graph():
    disorder = choose_disorder()

    if disorder is None:
        return

    col = disorders[disorder]

    yearly = df.groupby(year_col)[col].mean().dropna()

    if yearly.empty:
        print("No valid year data available.")
        return

    yearly.plot(
        kind="line", marker="o", figsize=(9, 5)
    )

    plt.title("Yearly Trend: " + disorder)
    plt.xlabel("Year")
    plt.ylabel("Average prevalence (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

##10.Trends of all Disorders
def all_trends_graph():
    plt.figure(figsize=(10, 6))

    for disorder, col in disorders.items():
        yearly = df.groupby(year_col)[col].mean().dropna()

        plt.plot(
            yearly.index,
            yearly.values,
            label=disorder
        )

    plt.title("Mental Health Disorders Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Average prevalence (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

##11.Top 10 Entities
def top_countries_graph():
    disorder = choose_disorder()

    if disorder is None:
        return

    col = disorders[disorder]

    averages = df.groupby(entity_col)[col].mean().dropna()
    top = averages.nlargest(10)

    if top.empty:
        print("No valid data available.")
        return

    print("\nTop 10 entities for", disorder)
    print(top.round(4).to_string())

    top.sort_values().plot(
        kind="barh", figsize=(9, 6)
    )

    plt.title("Top 10 Entities: " + disorder)
    plt.xlabel("Average prevalence (%)")
    plt.ylabel("Country/entity")
    plt.tight_layout()
    plt.show()

##12.Lowest 10 Entities
def lowest_countries():
    disorder = choose_disorder()

    if disorder is None:
        return

    col = disorders[disorder]

    averages = df.groupby(entity_col)[col].mean().dropna()
    lowest = averages.nsmallest(10)

    print("\nLowest 10 entities for", disorder)
    print(lowest.round(4).to_string())

##13.India Analysis
def india_analysis():
    india = df[
        df[entity_col].astype(str).str.strip().str.lower()
        == "india"
    ].copy()

    if india.empty:
        print("India was not found in the dataset.")
        return

    india = india.sort_values(year_col)

    print("\n========== INDIA ANALYSIS ==========")
    print("Number of records:", len(india))

    print("\nAverage prevalence in India:")

    for disorder, col in disorders.items():
        print(disorder + ":", round(india[col].mean(), 4))

    print("\nIndia data:")
    print(
        india[[entity_col, year_col] + list(disorders.values())]
        .head()
        .to_string(index=False)
    )

    india = india.set_index(year_col)

    india[list(disorders.values())].plot(
        figsize=(10, 6)
    )

    plt.title("Mental Health Trends in India")
    plt.xlabel("Year")
    plt.ylabel("Prevalence (%)")
    plt.legend(list(disorders.keys()))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

## 14.Histogram
def histogram_graph():
    disorder = choose_disorder()

    if disorder is None:
        return

    values = df[disorders[disorder]].dropna()

    if values.empty:
        print("No valid data available.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=15, edgecolor="black")

    plt.title("Distribution of " + disorder)
    plt.xlabel("Prevalence (%)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.show()

##15.Boxplot
def boxplot_graph():
    plot_data = df[list(disorders.values())].copy()
    plot_data.columns = list(disorders.keys())

    plot_data.plot(kind="box", figsize=(9, 5))

    plt.title("Distribution of Mental Health Disorders")
    plt.ylabel("Prevalence (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

## 16.Scatter Plot
def scatter_graph():
    names = list(disorders.keys())

    print("\nAvailable disorders:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")

    try:
        first = int(input("Choose first disorder: "))
        second = int(input("Choose second disorder: "))

        if not (
            1 <= first <= len(names)
            and 1 <= second <= len(names)
        ):
            print("Invalid choice.")
            return

    except ValueError:
        print("Enter numbers only.")
        return

    x_name = names[first - 1]
    y_name = names[second - 1]

    plot_data = df[
        [disorders[x_name], disorders[y_name]]
    ].dropna()

    if plot_data.empty:
        print("No valid data available.")
        return

    plt.figure(figsize=(8, 5))

    plt.scatter(
        plot_data.iloc[:, 0],
        plot_data.iloc[:, 1],
        alpha=0.5
    )

    plt.title(x_name + " vs " + y_name)
    plt.xlabel(x_name + " prevalence (%)")
    plt.ylabel(y_name + " prevalence (%)")
    plt.tight_layout()
    plt.show()

##17.Correlation Heatmap
def correlation_graph():
    plot_data = df[list(disorders.values())].copy()
    plot_data.columns = list(disorders.keys())

    correlation = plot_data.corr()

    plt.figure(figsize=(8, 6))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            value = correlation.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j, i, round(value, 2),
                    ha="center", va="center"
                )

    plt.title("Correlation Between Mental Health Disorders")
    plt.tight_layout()
    plt.show()

## 18.Main Menu
while True:
    print("\n======================================")
    print(" MENTAL HEALTH ANALYSIS SYSTEM")
    print("======================================")

    print("1. Dataset and statistical summary")
    print("2. Data cleaning checks")
    print("3. Average prevalence bar graph")
    print("4. Horizontal bar graph")
    print("5. Yearly trend of one disorder")
    print("6. Trends of all disorders")
    print("7. Top 10 countries/entities")
    print("8. Lowest 10 countries/entities")
    print("9. India analysis")
    print("10. Histogram")
    print("11. Box plot")
    print("12. Scatter plot")
    print("13. Correlation heatmap")
    print("0. Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        show_summary()

    elif choice == "2":
        clean_data()

    elif choice == "3":
        average_disorder_graph()

    elif choice == "4":
        horizontal_bar_graph()

    elif choice == "5":
        trend_graph()

    elif choice == "6":
        all_trends_graph()

    elif choice == "7":
        top_countries_graph()

    elif choice == "8":
        lowest_countries()

    elif choice == "9":
        india_analysis()

    elif choice == "10":
        histogram_graph()

    elif choice == "11":
        boxplot_graph()

    elif choice == "12":
        scatter_graph()

    elif choice == "13":
        correlation_graph()

    elif choice == "0":
        print("\nThank you for using the program!")
        break

    else:
        print("Invalid choice. Please try again.")