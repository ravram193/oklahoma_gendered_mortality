import pandas as pd
import numpy as np

def clean_dataframe(df, sex):

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

if __name__ == "__main__":

    # Upload raw data (taken from https://www.health.state.ok.us/stats/Vital_Statistics/Death/Final/Statistics21Trend.shtml)
    deaths_male = pd.read_csv('../data/oklahoma_deaths_male.csv')
    deaths_female = pd.read_csv('../data/oklahoma_deaths_female.csv')

    # Cleaning raw datasets for male and female deaths
    deaths_male_cleaned = clean_dataframe(deaths_male, "Male")
    deaths_female_cleaned = clean_dataframe(deaths_female, "Female")

    # Merging into one DataFrame
    all_deaths_ok = pd.concat([deaths_male_cleaned,deaths_female_cleaned])

    # Saving
    all_deaths_ok.to_csv("../data/all-deaths_by-county-sex-race_oklahoma_2010-2023.csv", index=False)