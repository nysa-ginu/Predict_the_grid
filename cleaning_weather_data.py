'''
Purpose of this module is to create one csv accumulating all the data present in the weather_data folder.
'''

from pyspark.sql import SparkSession
from pyspark.sql.functions import *


spark = SparkSession.builder.getOrCreate()

json_files = "weather_data/*.json"

# Read the JSON files into a DataFrame
df = spark.read.json(json_files)


# Select the desired fields
selected_fields = [
    col("lat").alias("lat"),
    col("lon").alias("lng"),
    col("data").getItem(0).dt.alias("time_unix"),
    col("data").getItem(0).temp.alias("temp"),
    col("data").getItem(0).pressure.alias("pressure"),
    col("data").getItem(0).humidity.alias("humidity"),
    col("data").getItem(0).wind_speed.alias("wind_speed"),
    col("data").getItem(0).clouds.alias("clouds"),
    col("data").getItem(0).weather.getItem(0).main.alias("weather_desc")
]

df_selected = df.select(*selected_fields)
df_selected = df_selected.withColumn("normal_time", to_timestamp("time_unix"))
df_selected = df_selected.withColumn("date", split(df_selected["normal_time"], " ").getItem(0))
df_selected = df_selected.withColumn("time", split(df_selected["normal_time"], " ").getItem(1))
df_selected = df_selected.coalesce(1)

#----------------------------------------Combining weather data with others-------------------------

'''
The date for each race is unique. Therefore, we will use that to get other info from races.csv
'''

df2 = spark.read.csv("raw_data/races.csv", header=True, inferSchema=True)

# Select specific columns from each DataFrame
df1_selected = df_selected.select("temp", "pressure", "humidity", "wind_speed", "clouds", "weather_desc", "date")
df2_selected = df2.select("raceId", "year", "circuitId", "name", "date")

# Merge DataFrames on a common column
merged_df = df1_selected.join(df2_selected, on="date", how="left")

# Sort the DataFrame based on two columns in ascending order
sorted_df = merged_df.orderBy(col("year"), col("name"))

#rearranging columns

desired_order = ['raceId', 'circuitId', 'year', 'date', 'name', 'temp', 
                'pressure', 'humidity', 'wind_speed', 'clouds', 'weather_desc']

sorted_df = sorted_df.select(desired_order)

# Show the resulting DataFrame
sorted_df.show()
sorted_df = sorted_df.coalesce(1)

# Save the DataFrame as a single CSV file
csv_path = "weather.csv"
sorted_df.write.csv(csv_path, header=True, mode="overwrite")
