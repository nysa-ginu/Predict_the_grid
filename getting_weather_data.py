'''
PLAN OF ACTION:

get data from the V8 era onwards (2006 onwards).
-   First we will scrap the f1-fansite for the weather data. <------ dropping this because it gave 403 error. forbidden page. We should respect the policies of the web page.
-   To fill the gaps where the data is missing from the f1-fansite,
    we will use the OpenWeatherMap API for this.

For this, the attributes we need are:
-   year (Can be manually entered)
-   Grand Prix names of each year (can be found from races.csv)
-   latitude of each grand prix location (can be found from circuits.csv by linking with races.csv through circuitID)
-   longitude of each grand prix location (can be found from circuits.csv by linking with races.csv through circuitID)
-   date : date of the race
'''


import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import date, datetime

def is_date_before_today(target_date):
    '''
    This function is to check if the date of the grand_prix is before today
    to make sure it has been conducted.
    '''

    target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    current_date = date.today()
    return target_date < current_date

api_key = 'YOUR API KEY'
base_url = 'https://api.openweathermap.org/data/3.0/onecall/timemachine'

def get_timestamp(date_gp, time):

    date_time = date_gp + ' ' + time
    dt = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    timestamp = int(dt.timestamp())
    print(timestamp)

    return timestamp

def get_weather_data(lat, lng, date_gp, time, filename):

    timestamp_fin = get_timestamp(date_gp, time)

    api_key = 'YOUR API KEY'
    api_endpoint = 'http://api.openweathermap.org/data/3.0/onecall/timemachine'
    params = {
        'lat': lat,
        'lon': lng,
        'dt': timestamp_fin,
        'appid': api_key,
    }
    response = requests.get(api_endpoint, params=params)
    response.raise_for_status()

    if response.status_code == 200:
        data = response.json()

        #storing the json files
        with open(filename, 'w') as file:
            json.dump(data, file)

        print("Historical weather data saved successfully for " , filename)

    else:
        print('Failed to retrieve weather data:', response.status_code)
    


# data needed

df_races = pd.read_csv('raw_data/races.csv')
df_circuits = pd.read_csv('raw_data/circuits.csv')

#----------------------------final dataset structure-------------------------------------------------------

df_final = pd.DataFrame()

condition = df_races['year'] >= 2006
df_final['year'] = df_races.loc[condition, 'year']
df_final['name'] = df_races.loc[condition, 'name']
df_final['date'] = df_races.loc[condition, 'date']
df_final['time'] = df_races.loc[condition, 'time']
df_final['circuitId'] = df_races.loc[condition, 'circuitId']


df_merged = pd.merge(df_final, df_circuits[['circuitId', 'lat', 'lng']], on='circuitId', how='left')

df_merged = df_merged.drop_duplicates(subset= ['year', 'name']).reset_index(drop=True)

df_merged = df_merged.sort_values(['year', 'name'])


#--------------------------USING OPENWEATHERMAP API FOR WEATHER CONDITIONS-------------------------------------------

for index, row in df_merged.iterrows():

    flag = False

    lat = row['lat']
    lng = row['lng']
    date_gp = row['date']
    time = row['time']

    filename = str(row['year']) + "_" + str(row['name']) + ".json"
    file_name = str('weather_data/') + str(row['year']) + "_" + str(row['name']) + ".json"

    #checking if the file exists
    file_path = 'weather_data/'
    files = os.listdir(file_path)

    for file in files:
        if file == filename:
            flag = True

    if flag == False: 
        if is_date_before_today(date_gp):
            print(row)
            get_weather_data(lat, lng, date_gp, time, file_name)
