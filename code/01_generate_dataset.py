import pandas as pd
import numpy as np

def clean_deaths_dataframe(df, sex):

    """Cleans the raw Oklahoma deaths DataFrame for a given sex.
    
    Args: 
    - df (pd.DataFrame): Raw DataFrame to be cleaned. 
    - sex (str): Sex category for the data ("Male" or "Female"). This will be added as a new column.

    Returns:       
    - tidy (pd.DataFrame): Cleaned and tidy DataFrame with columns: year, county, sex, race, deaths.

    """

    # Rename raw columns
    df = df.rename(columns={
        df.columns[0]: "col0", # Search Characteristic
        df.columns[1]: "col1", # Values Selected
        df.columns[2]: "col2"  # Unnamed: 2 (Deaths)
    })

    # Extract YEAR (this must happen BEFORE filtering)
    df["year"] = df["col0"].astype(str).str.extract(r"(\d{4})")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["year"] = df["year"].ffill() # Now year should populate for all rows

    # Race list for classification
    race_list = [
        "White",
        "Black or African American",
        "American Indian or Alaska Native",
        "Asian",
        "Native Hawaiian or Other Pacific Islander",
        "Other",
        "Unknown",
        "More than one race"
    ]

    # Identify COUNTY rows
    def is_county(x):
        if pd.isna(x): 
            return False
        x = str(x).strip()
        if x in race_list:
            return False
        if x in ["County of Residence", "Search Characteristic", "Values Selected"]:
            return False
        if x.startswith("Year"):
            return False
        return True

    df["county_raw"] = df["col0"].where(df["col0"].apply(is_county))
    df["county"] = df["county_raw"].ffill()

    # Extract RACE
    df["race"] = df["col1"].where(df["col1"].isin(race_list))

    # Extract DEATHS
    df["deaths"] = df["col2"].replace(".", np.nan)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df['deaths'] = df['deaths'].fillna(0)

    # Keep only race rows
    df = df[df["race"].isin(race_list)]

    # Categorizing as male or female
    df['sex'] = sex

    # Final tidy format
    tidy = df[["year", "county", "sex", "race", "deaths"]].reset_index(drop=True)

    return tidy

def clean_birth_rate_dataframe(df, demo_col, demo_list):

    """Cleans a raw Oklahoma births dataset, creating sub-divisions for year, county, and a third specified demographic category.
    
    Args: 
    - df (pd.DataFrame): Raw DataFrame to be cleaned. Must include birth data by year, county, and an additional demographic variable.
    - demo_col (str): Name of a new column for the additional demographic category included in the dataset.
    - demo_list (list): List of categories for the specified demographic group.

    Returns:       
    - tidy (pd.DataFrame): Cleaned and tidy DataFrame with columns: year, county, `demo_col`, birth, population, and birth rate.

    """

    # Rename raw columns
    df = df.rename(columns={
        df.columns[0]: "col0", # Search Characteristic
        df.columns[1]: "col1", # Values Selected
        df.columns[2]: "col2", # Unnamed: 2 (Births)
        df.columns[3]: "col3", # Unnamed: 3 (Population)
        df.columns[4]: "col4"  # Unnamed: 4 (Birth Rate)
    })

    # Extract YEAR (this must happen BEFORE filtering)
    df["year"] = df["col0"].astype(str).str.extract(r"(\d{4})")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["year"] = df["year"].ffill() # Now year should populate for all rows

    # Identify COUNTY rows
    def is_county(x):
        if pd.isna(x): 
            return False
        x = str(x).strip()
        if x in demo_list:
            return False
        if x in ["County of Residence", "Search Characteristic", "Values Selected"]:
            return False
        if x.startswith("Year"):
            return False
        return True

    df["county_raw"] = df["col0"].where(df["col0"].apply(is_county))
    df["county"] = df["county_raw"].ffill()

    # Extract values for each demographic sub-group
    df[demo_col] = df["col1"].where(df["col1"].isin(demo_list))

    # Extract births, population, and birth rate
    df["births"] = df["col2"].replace(".", np.nan)
    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    df["births"] = df["births"].fillna(0)
    df["population"] = df["col3"].replace(".", np.nan)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["population"] = df["population"].fillna(0)
    df["birth_rate"] = df["col4"].replace(".", np.nan)
    df["birth_rate"] = pd.to_numeric(df["birth_rate"], errors="coerce")
    df["birth_rate"] = df["birth_rate"].fillna(0)

    # Keep only rows related to the specified demographic group
    df = df[df[demo_col].isin(demo_list)]

    # Final tidy format
    tidy = df[["year", "county", demo_col, "births", "population", "birth_rate"]].reset_index(drop=True)

    return tidy

def clean_live_births_dataframe(df, demo_col, demo_list):
    """Cleans a raw Oklahoma births dataset, creating sub-divisions for year, county (optional), and a specified demographic category.
    Args: 
    - df (pd.DataFrame): Raw DataFrame to be cleaned. Must include birth data by year, live births, % live births, and an additional demographic variable.
    - demo_col (str): Name of a new column for the additional demographic category included in the dataset.
    - demo_list (list): List of categories for the specified demographic group.
    Returns:       
    - tidy (pd.DataFrame): Cleaned and tidy DataFrame with columns: year, county (if present), `demo_col`, live births, and % live births.
    """
    
    # Rename raw columns
    df = df.rename(columns={
        df.columns[0]: "col0",
        df.columns[1]: "col1",
        df.columns[2]: "col2",
        df.columns[3]: "col3",
    })

    # Extract YEAR
    df["year"] = df["col0"].astype(str).str.extract(r"(\d{4})")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["year"] = df["year"].ffill()

    # Identify COUNTY rows
    def is_county(x):
        if pd.isna(x):
            return False
        x = str(x).strip()
        if x in demo_list:
            return False
        if x in ["County of Residence", "Search Characteristic", "Values Selected", "Mother's Education-5 Category"]:
            return False
        if x.startswith("Year"):
            return False
        return True

    df["county_raw"] = df["col0"].where(df["col0"].apply(is_county))
    df["county"] = df["county_raw"].ffill().str.strip()

    # Extract demographic variable
    df[demo_col] = df["col0"].where(df["col0"].isin(demo_list))
    df[demo_col] = df[demo_col].fillna(df["col1"].where(df["col1"].isin(demo_list)))

    # Extract births and percent live births
    df["live_births"] = pd.to_numeric(df["col2"].replace(".", np.nan), errors="coerce").fillna(0)
    df["percent_live_births"] = pd.to_numeric(df["col3"].replace(".", np.nan), errors="coerce").fillna(0)

    # Keep only rows related to the specified demographic group
    df = df[df[demo_col].isin(demo_list)]

    # Create full combination of year × county × demographic category
    years = df["year"].dropna().unique()
    counties = df["county"].dropna().unique()
    full_index = pd.MultiIndex.from_product(
        [years, counties, demo_list],
        names=["year", "county", demo_col]
    )
    df_full = pd.DataFrame(index=full_index).reset_index()

    # Merge with actual data
    tidy = pd.merge(df_full, df[["year", "county", demo_col, "live_births", "percent_live_births"]],
                    on=["year", "county", demo_col], how="left")

    # Fill missing values with 0
    tidy["live_births"] = tidy["live_births"].fillna(0)
    tidy["percent_live_births"] = tidy["percent_live_births"].fillna(0)

    # Aggregate duplicates
    tidy = tidy.groupby(["year", "county", demo_col], as_index=False).agg({
        "live_births": "sum",
        "percent_live_births": "mean"
    })

    return tidy

if __name__ == "__main__":

    # # Upload raw data (taken from https://www.health.state.ok.us/stats/Vital_Statistics/Death/Final/Statistics21Trend.shtml)
    # deaths_male = pd.read_csv('../data/input/oklahoma_deaths_male.csv')
    # deaths_female = pd.read_csv('../data/input/oklahoma_deaths_female.csv')

    # # Cleaning raw datasets for male and female deaths
    # deaths_male_cleaned = clean_deaths_dataframe(deaths_male, "Male")
    # deaths_female_cleaned = clean_deaths_dataframe(deaths_female, "Female")

    # # Merging into one DataFrame
    # all_deaths_ok = pd.concat([deaths_male_cleaned,deaths_female_cleaned])

    # # Saving
    # all_deaths_ok.to_csv("../data/output/all-deaths_by-county-sex-race_oklahoma_2010-2023.csv", index=False)

    # BIRTHS BY RACE

    # Upload raw data (taken from https://www.health.state.ok.us/stats/Vital_Statistics/Birth/Final/Statistics_2021upTrend.shtml)
    births_by_race = pd.read_csv('../data/input/oklahoma_births-by-county_by-race_2010-2024.csv')

    # Race list for classification
    race_list = [
        "White",
        "Black or African American",
        "American Indian or Alaska Native",
        "Asian",
        "Native Hawaiian or Other Pacific Islander",
        "Other",
        "Unknown",
        "More than one race"
    ]

    # Cleaning raw dataset for births by race
    births_by_race_cleaned = clean_birth_rate_dataframe(births_by_race, "race", race_list)

    # Saving
    births_by_race_cleaned.to_csv("../data/output/oklahoma_births-by-county_by-race_2010-2024.csv", index=False)

    # BIRTHS BY AGE

    # Upload raw data
    births_by_age = pd.read_csv('../data/input/oklahoma_births_by-county_by-age_2010-2024.csv')

    # Age list for classification
    age_list = [
        "10-14 years",
        "15-17 years",
        "18-19 years",
        "20-24 years",
        "25-29 years",
        "30-34 years",
        "35-39 years",
        "40-44 years",
        "45-54 years",
        "Unknown Age"
    ]

    # Cleaning raw dataset for births by race
    births_by_age_cleaned = clean_birth_rate_dataframe(births_by_age, "age", age_list)

    # Saving
    births_by_age_cleaned.to_csv("../data/output/oklahoma_births-by-county_by-age_2010-2024.csv", index=False)

    # LIVE BIRTHS BY GESTATIONAL AGE

    births_by_gestational_age = pd.read_csv('../data/input/oklahoma_birth-weight_2010-2024.csv')

    gestation_age_list = [
        '<32 weeks', 
        '32-36 weeks', 
        '37-39 weeks', 
        '40-41 weeks', 
        '42+ weeks', 
        'Unknown'
        ]
    
    births_by_gestational_age_cleaned = clean_live_births_dataframe(births_by_gestational_age, 'gestation_age', gestation_age_list)

    # Saving
    births_by_gestational_age_cleaned.to_csv("../data/output/oklahoma_birth-weight_2010-2024.csv", index=False)

    # LIVE BIRTHS BY GESTATIONAL AGE

    births_by_education = pd.read_csv('../data/input/oklahoma_births_by-mothers-edu_2010-2024.csv')

    education_list = [
        "0 - 8 years",
        "9 - 11 years",
        "12 years",
        "13 - 15 years",
        ">= 16 years",
        "UNKNOWN"
        ]
    
    births_by_education_cleaned = clean_live_births_dataframe(births_by_education, 'education', education_list)

    # Saving
    births_by_education_cleaned.to_csv("../data/output/oklahoma_births_by-mothers-edu_2010-2024.csv", index=False)