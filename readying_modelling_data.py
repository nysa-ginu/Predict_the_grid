'''

The purpose of this module is to get the data ready for modelling which includes data integration from th raw_data
folder; cleaning and converting it into suitable format for modelling.

'''

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
import numpy as np


def find_correlation(correlation_matrix):
    # Set your desired correlation threshold
    correlation_threshold = 0.8

    # Find the highly correlated columns (positive correlation)
    highly_correlated_positive = correlation_matrix[
        (correlation_matrix > correlation_threshold) & (correlation_matrix < 1.0)
    ].dropna(how='all').dropna(axis=1, how='all').stack().index.tolist()

    # Find the highly correlated columns (negative correlation)
    highly_correlated_negative = correlation_matrix[
        (correlation_matrix < -correlation_threshold) & (correlation_matrix > -1.0)
    ].dropna(how='all').dropna(axis=1, how='all').stack().index.tolist()

    # Print the highly correlated columns (positive correlation)
    print("Highly correlated columns (positive correlation):")
    for col1, col2 in highly_correlated_positive:
        print(col1, "and", col2)

    # Print the highly correlated columns (negative correlation)
    print("Highly correlated columns (negative correlation):")
    for col1, col2 in highly_correlated_negative:
        print(col1, "and", col2)



def get_train_test_set():


    spark = SparkSession.builder.getOrCreate()


    #loading dataframes
    df_const = spark.read.csv("raw_data/constructors.csv", header=True, inferSchema=True)
    df_drivers = spark.read.csv("raw_data/drivers.csv", header=True, inferSchema=True)
    df_quali = spark.read.csv("raw_data/qualifying.csv", header=True, inferSchema=True)
    df_races = spark.read.csv("raw_data/races.csv", header=True, inferSchema=True)
    df_results = spark.read.csv("raw_data/results.csv", header=True, inferSchema=True)
    df_weather = spark.read.csv("raw_data/weather.csv", header=True, inferSchema=True)

    #selecting the desired attributes
    df_const_selected = df_const.select("constructorId", "name")
    df_drivers_selected = df_drivers.select("driverId", "code", "dob")
    df_quali_selected = df_quali.select("raceId", "driverId", "constructorId", "q1", "q2" ,"q3")
    df_races_selected = df_races.select("raceId", "year", "round", "name", "date")
    df_results_selected = df_results.select("raceId", "driverId", "constructorId", "grid", "position")
    df_weather_selected = df_weather.select("raceId", "temp", "pressure", "humidity", "wind_speed", "clouds", "weather_desc")

    '''
    rename a few columns 
        - "df_const_selected.name" -> "df_const_selected.constructorName"
        - "df_drivers_selected.code" -> "df_drivers_selected.driverCode"
        - "df_races_selected.name" -> "df_races_selected.GPName"
        - "df_results_selected.grid" -> "df_results_selected.gridPosition"
        - "df_results_selected.position" -> "df_results_selected.finalPosition"
    '''

    df_const_selected = df_const_selected.withColumnRenamed("name","constructorName")
    df_drivers_selected = df_drivers_selected.withColumnRenamed("code","driverCode")
    df_races_selected = df_races_selected.withColumnRenamed("name","GPName")
    df_results_selected = df_results_selected.withColumnRenamed("grid","gridPosition") \
                                            .withColumnRenamed("position", "finalPosition")


    '''

    FURTHER PLAN OF ACTION:
    - base dataset = df_results_selected
    -JOINS:
        - w/ df_const_selected : (on="constructorId", how="left")
        - w/ df_drivers_selected : (on="driverId", how="left")
        - w/ df_quali_selected : (on= "raceId", "driverId", "constructorId", how="left")
        - w/ df_races_selected : (on="raceId", how="left")
        - w/ df_status: (on="statusId", how="left")
        - w/ df_weather_selected : (on="raceId", how="left")

    '''

    quali_join_columns = ["raceId", "driverId", "constructorId"]
    joined_df = df_results_selected.join(df_const_selected, "constructorId", "left")
    joined_df = joined_df.join(df_drivers_selected, "driverId", "left")
    joined_df = joined_df.join(df_quali_selected, quali_join_columns, "left")
    joined_df = joined_df.join(df_races_selected, "raceId", "left")
    joined_df = joined_df.join(df_weather_selected, "raceId", "left")

    #get data from the V8-era onwards
    joined_df = joined_df.filter(joined_df["year"] >= 2006)
    joined_df = joined_df.orderBy(col("year"), col("GPName"))

    '''
    Converting the final dataframe to pandas dataframe; handling nan values and converting minutes:seconds.milliseconds to ms
    '''

    joined_df_pandas = joined_df.toPandas()
    joined_df_pandas = joined_df_pandas.replace('\\N', np.nan)
    joined_df_pandas['q1'] = pd.to_timedelta('00:' + joined_df_pandas['q1']).dt.total_seconds() * 1000
    joined_df_pandas['q2'] = pd.to_timedelta('00:' + joined_df_pandas['q2']).dt.total_seconds() * 1000
    joined_df_pandas['q3'] = pd.to_timedelta('00:' + joined_df_pandas['q3']).dt.total_seconds() * 1000

    '''
    Calculating the age of the drivers at the time of the race. Age has played a major factor in 
    getting the a psotion on the grid for the drivers. The younger you are, your performance will be better, is what is considered.
    But recently with Fernando Alonso being one of the oldest drivers on the grid and yet is at the top of his game, beating 
    drivers much younger than him, this whole concept of the "age" has been brought into question. 
    '''

    joined_df_pandas['dob'] = pd.to_datetime(joined_df_pandas['dob'])
    joined_df_pandas['date'] = pd.to_datetime(joined_df_pandas['date'])
    joined_df_pandas['age'] = (joined_df_pandas['date'] - joined_df_pandas['dob']).dt.days // 365
    joined_df_pandas.drop(['dob', 'date'], axis=1, inplace=True)

    '''
    Handling cases that DNF'd or DNS'd because they have a missing finalPosition value.
    '''

    joined_df_pandas.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing = joined_df_pandas['finalPosition'].isna()
    missing_cumsum = missing.cumsum()
    offset = missing_cumsum - missing_cumsum.where(~missing).ffill().fillna(0)
    offset = offset.astype(int)
    joined_df_pandas['finalPosition'] = joined_df_pandas['finalPosition'].ffill().astype(int)
    joined_df_pandas['finalPosition'] = joined_df_pandas['finalPosition'] + offset

    '''
    Handling case where the driver might have gotten out in Q1 or Q2 or DNQ'd
    '''

    # Forward fill missing values in q1
    joined_df_pandas['q1'].fillna(method='ffill', inplace=True)

    # Replace NaN values in q1 with previous non-missing value + 1
    joined_df_pandas['q1'].fillna(joined_df_pandas['q1'].shift() + 1, inplace=True)

    # Replace NaN values in q2 and q3 with values from q1
    joined_df_pandas['q2'].fillna(joined_df_pandas['q1'], inplace=True)
    joined_df_pandas['q3'].fillna(joined_df_pandas['q1'], inplace=True)


    '''
    Feature Engineering (One-Hot encoding, Correlation)
    '''

    #--------------------------ONE HOT ENCODING--------------------------------------------------


    one_hot_col = ['constructorName', 'GPName', 'weather_desc']

    one_hot_encoded_data = pd.get_dummies(joined_df_pandas, columns = one_hot_col)

    #----------------------------------dropping ID columns----------------------------------------

    one_hot_encoded_data.drop(['raceId', 'driverId', 'constructorId'], axis=1, inplace=True)


    #--------------------finding highly correlated columns (negatively and positively)-------------------------

    correlation_matrix = one_hot_encoded_data.corr()
    find_correlation(correlation_matrix)


    '''
    Creating train, validation, test sets

    train data : 2006 - 2021
    validation data : 2022

    '''

    train_data = one_hot_encoded_data[one_hot_encoded_data['year'] <= 2021]
    val_data = one_hot_encoded_data[one_hot_encoded_data['year'] ==2022 ]

    return one_hot_encoded_data, train_data, val_data