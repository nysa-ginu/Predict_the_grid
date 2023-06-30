import pandas as pd
import requests
from bs4 import BeautifulSoup



def get_proper_gp_name_4_web_scraping(gp_name):

    '''
    function converts the name of the grandprix into a web scraping friendly one

    eg: 
    input : Australian Grand Prix
    output : australian

    '''
    name  = gp_name.rsplit(' ', 2)[0]
    if len(name) == 2:
        name = '-'.join(name)
    else:
        name = name
    name = name.lower()

    return name


def scrape_weather_data(year, grand_prix):

    '''
    function performas web-scarping to get the weather information

    arguments:
              year : year of the grand prix
              grand-prix : name of the grand prix
            
    returns:
            weather : weather description
            temperature  : temperature during the time of the race
    '''

    urls = ['https://www.f1-fansite.com/f1-result/race-results-{year}-{grand_prix}-f1-grand-prix/',
               'https://www.f1-fansite.com/f1-result/race-results-{year}-{grand_prix}-f1-gp/',
               'https://www.f1-fansite.com/f1-result/race-result-{year}-{grand_prix}-f1-gp/',
               'https://www.f1-fansite.com/f1-result/{year}-{grand_prix}-grand-prix-race-results/',
               'https://www.f1-fansite.com/f1-result/{year}-{grand_prix}-grand-prix-results/']

    for url in urls:

        # Send a GET request to the website
        response = requests.get(url)
        #if the page does not exists, then skip
        if response.status_code == 404:
            print(f"URL {url} returned a 404 error. Skipping...")
            continue

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the elements containing weather and temperature data
        weather_element = soup.find("p", text="Weather: ")
        temperature_element = soup.find("p", text="Air temp: ")

        # Extract the weather and temperature values
        weather = weather_element.text if weather_element else None
        temperature = temperature_element.text if temperature_element else None

    return weather, temperature