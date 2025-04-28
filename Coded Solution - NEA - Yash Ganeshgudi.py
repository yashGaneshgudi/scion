#We install pandas to store the dataframes. Dataframes are used to store our weather and commodity price datasets.
import pandas as pd 
import numpy as np

#CSV is imported as we are also working with .CSV files and pandas dataframes interchangeably.
import csv

#Sklearn is an open-source machine learning library in python and is what we use within our program to implement an SVM.
from sklearn.svm import SVC

#We import a scale module from sklearn to make the weather(features) dataset have a mean of 0 and a standard deviation of 1.
from sklearn.preprocessing import scale 

#Grid-search cross validation module is also used to do the grid search cross validation if our program doesn't do it manually.
from sklearn.model_selection import GridSearchCV



#Datatime is used to handle dates, they are all in the form ISO8601 so can be easily accessed and understood.
from datetime import datetime, timedelta

#These are modules required by openMeteo as we are accessing an external database through an API.
import openmeteo_requests
#These two modules are used to retry connecting to an API if the user disconnects from the server.
import requests_cache
from retry_requests import retry

#Web scraping modules are included here beautifulsoup4 is used to retrieve the HTML code and requests to access elements within a website.
from bs4 import BeautifulSoup
import requests

#Once we have the link to an external dataset we will use os to download it.
import os

#When we implemented a loading screen we used subprocess to run multiple threads of code at once.
import subprocess

#Abstract syntax tree used for the loading screen.
import ast

import time#Allows us to sequence and syncronise our functions
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox 

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

#Added support for advanced error handling to filter through and catch errors before they are shown to the user.
import warnings
import logging

global worldBank_url
worldBank_url = "https://www.worldbank.org/en/research/commodity-markets#1"

global cropIndex # This is a global crop index with all the possible crops we have available.
cropIndex = ['cocoa','soybean','corn','wheat','sugar']

#Suppresses logs about the systems CPU Architecture. e.g Apple Silicon(RISC)
logging.getLogger('IMK').setLevel(logging.WARNING)

# This is an error alert function and will be what is called when an error is caught within the program.
#This function allows the program to make decisions when errors occur.
def errorAlert(error,type):
    print('error encountered')
    print(error)

    #The type of the error defines the next steps to be taken after the error has been detected. 
    #In this case there are two types a soft and a hard reset. 
    #A soft reset will return back to the start page, whereas a hard reset will initialize the entire program.
    if type == 'soft reset':
        print('soft reset - returning to start page')
    else:
        type == 'hard reset'
        print('hard reset - resetting entire program')


# %% [markdown]
# Retrieve Historical Weather Data

# %%
# crop + Coordinates indicate the top three major producers of that crop around the world.errorAlert
# These three major producers are represented as three distinct coordinates with a latitude and a longitude component
#  
#The volume spread indicates what percentage of the produce that each location produces. 

cocoaCoordinates = ((6.053110308791517, -5.632365625597256),
                    (5.723258880120529, -2.0289499687551404),
                    (-1.4526661026550038, -80.30948687187554))
cocoaVolumeSpread = (0.660, 0.187, 0.153)

soybeanCoordinates = ((-19.64619384124217, -54.26514130936858),
                      (42.29404827977364, -92.81193554839746),
                      (-35.06747636366788, -58.071092994544266))
soybeanVolumeSpread = (0.487, 0.360, 0.153)

sugarCoordinates = ((-22.591058675268453, -48.380706513546365),
                    (27.512949904632368, 80.63046034857008),
                    (16.17804967961032, 103.56698522101857))
sugarVolumeSpread = (0.471, 0.404, 0.125)

wheatCoordinates = ((33.91053065183956, 113.67123697952496),
                    (47.904101854372406, 2.055730082984269),
                    (27.166528613792373, 80.59817535483508))
wheatVolumeSpread = (0.357, 0.353, 0.290)

cornCoordinates = ((42.685090443052125, -93.32174099559683),
                   (40.954928136698115, 117.03116790283333),
                   (-11.258015767551417, -54.79968906997016))
cornVolumeSpread = (0.474, 0.377, 0.149)

def verifyCoordinateValidity(latitude,longitude):#Returns validity of the coordinates
    try:
        if -90 < latitude < 90:
            validLatitude = True
        else:
            validLatitude = False #Invalid latitude. The value is not between -90 and 90
    except:
        validLatitude = False #Invalid input. The value is invalid as it is not a numerical value.
    
    try:
        if -180 < longitude < 180:
            validLongitude = True
        else:
            validLongitude = False #Invalid latitude. The value is not between -180 and 180
    except:
        validLongitude = False #Invalid input. The value is invalid as it is not a numerical value.
    
    if validLatitude == True and validLongitude == True:#The entire prorgam will return either aa valid / invalid statement that corresponds to the coordinate components.
        return 'valid values'
    else: 
        return 'invalid values'


def openMeteo_historical_climate(latitude, longitude):
    
    #This is a validation technique used to retry the connection constantly if the connection were to disconnect for any reason.
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://climate-api.open-meteo.com/v1/climate"#This is the url we use to directly access openMeteo's weather API.

    #Paramaters define data we input into the historical weather API.
    params = {
    #The location (coordinates) are expressed with the latitude and the longitude coordintates.
	"latitude": latitude,
	"longitude": longitude,

    #The start and end-date have been set to predetermined values.
    
	"start_date": "1960-01-01",
	"end_date": "2023-11-01",
	"models": "MRI_AGCM3_2_S",
	"daily": ["temperature_2m_mean", "cloud_cover_mean", "relative_humidity_2m_mean", "precipitation_sum"]
    }

    responses = openmeteo.weather_api(url, params=params)#We pass the parameters that we defined straight into and through the API.

    response = responses[0] # Process first location. Add a for-loop for multiple locations or weather models which we can do using multiple parameter tables.
    #Our implementation processes locations one at a time however all the data can theoreticallty be retrieved at once.
    #This is so we can clearly distinguish different pieces of data which would be difficult if it was all stored within a single dataframe.

    #These are identifiers which highlight the location and timezone of where the crop is grown.
    #These identifiers can be used to debug the program to find out what data is included within the dataframe.
    debugCoordinates = (f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    debugElevation = (f"Elevation {response.Elevation()} m asl")
    debugTimezone = (f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    debugTimezoneDifference = (f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process data in a daily format. By doing so we can extract the four metrics we are looking for.
    #These metrics are temperature, cloud_cover, humidity and precipitation
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_cloud_cover_mean = daily.Variables(1).ValuesAsNumpy()
    daily_relative_humidity_2m_mean = daily.Variables(2).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()

    #We need to generate a sequence of dates. To do this we repeat an interval between each date.
    daily_data = {"date": pd.date_range( 
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
    )}

    #These populate the dataframe with the necessary dates with the four metrics we just retrieved from the API.
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
    daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum

    daily_dataframe = pd.DataFrame(data = daily_data)

    #We return the entire historical weather dataset as a single panda's dataframe.
    return daily_dataframe

def retrieve_historical_climate(crop):#This entire function can retrieve all the historical weather data for a crop and accordingly merge all its datasets. 
    global historicalWeather1, historicalWeather2, historicalWeather3 #We make these dataframes have a global scope so they can be accessed across the entire program.

    #We use a for loop to loop through all three crop locations and extract the location and the respective coordinates.
    #This process occurs three times as we need to get the 1st, 2nd and 3rd historical weather dataset. This is because there are three major locations where each crop is grown. 
    for i in range (3): 
        latitude, longitude = globals()[crop + 'Coordinates'][i] #Using globals() instead of using multiple if statements for each crop we can write efficient code. This means that we can directly call the crop's coordinates.
        coordinateValidity = verifyCoordinateValidity(latitude, longitude)#We use validation to verify that the coordinates are correct and valid. 

        if coordinateValidity == 'valid values':#If the specific coordinates are valid we can generate each historical weather dataframe one at a time.
            globals()['historicalWeather' + str(i + 1)] = openMeteo_historical_climate(latitude, longitude) #Generates 3 different dataframes, one for each country

        #If data is incorrect we catch it before the error is allowed to propogate and call the error function.
        elif coordinateValidity == 'invalid values':
            errorAlert('Crop information database contains invalid longitude and latitude coordinates.','hard reset')
        else:
            errorAlert('Coordinates could not be validated. No data has been requested from the API OpenMeteo.','hard reset')
    
    historicalWeather1.to_csv('historicalWeather1.csv', sep=',', index=False, encoding='utf-8') #Saves the dataset as a .csv file. This allows for physical debugging as we have can see the .csv file in human read-able format as a csv is tabular.
    historicalWeather2.to_csv('historicalWeather2.csv', sep=',', index=False, encoding='utf-8') #.csv files and dataframes are interchangeable so it is efficient to save all weather datasets as CSVs.
    historicalWeather3.to_csv('historicalWeather3.csv', sep=',', index=False, encoding='utf-8') 

    #Converts .csv back into a pandas dataframe. This uses the utf-8 encoding.
    historicalWeather1 = pd.read_csv("historicalWeather1.csv", header=0)
    historicalWeather2 = pd.read_csv("historicalWeather2.csv", header=0)
    historicalWeather3 = pd.read_csv("historicalWeather3.csv", header=0)

    #This is where we manually multiply all the datasets with the respective percentage volume which that location produces.
    #We multiple every collumn within the dataset by the respective volume spread that location produces.
    # 
    #This is possible due to the consistent indexing and variable naming scheme within our program as the index of the coordinate is equal to the index of the percentage spread.
    #     
    historicalWeather1["temperature_2m_mean"] = (historicalWeather1["temperature_2m_mean"] * globals()[crop + 'VolumeSpread'][0])
    historicalWeather1["cloud_cover_mean"] = (historicalWeather1["cloud_cover_mean"] * globals()[crop + 'VolumeSpread'][0])
    historicalWeather1["relative_humidity_2m_mean"] = (historicalWeather1["relative_humidity_2m_mean"] * globals()[crop + 'VolumeSpread'][0])
    historicalWeather1["precipitation_sum"] = (historicalWeather1["precipitation_sum"] * globals()[crop + 'VolumeSpread'][0])

    historicalWeather2["temperature_2m_mean"] = (historicalWeather2["temperature_2m_mean"] * globals()[crop + 'VolumeSpread'][1])
    historicalWeather2["cloud_cover_mean"] = (historicalWeather2["cloud_cover_mean"] * globals()[crop + 'VolumeSpread'][1])
    historicalWeather2["relative_humidity_2m_mean"] = (historicalWeather2["relative_humidity_2m_mean"] * globals()[crop + 'VolumeSpread'][1])
    historicalWeather2["precipitation_sum"] = (historicalWeather2["precipitation_sum"] * globals()[crop + 'VolumeSpread'][1])

    historicalWeather3["temperature_2m_mean"] = (historicalWeather3["temperature_2m_mean"] * globals()[crop + 'VolumeSpread'][2])
    historicalWeather3["cloud_cover_mean"] = (historicalWeather3["cloud_cover_mean"] * globals()[crop + 'VolumeSpread'][2])
    historicalWeather3["relative_humidity_2m_mean"] = (historicalWeather3["relative_humidity_2m_mean"] * globals()[crop + 'VolumeSpread'][2])
    historicalWeather3["precipitation_sum"] = (historicalWeather3["precipitation_sum"] * globals()[crop + 'VolumeSpread'][2])

    #As all the historical weather dataframes have been processed we can just add them using an addition function which adds the respective numbers.
    combined_dataframe = historicalWeather1 + historicalWeather2 + historicalWeather3

    #The date which is a string has also been concatenated so we need to fix that.
    combined_dataframe["date"] = historicalWeather1["date"]

    #Let's format the data into yearly averages as we have too many daily weather datapoints.
    combined_dataframe["date"] = pd.to_datetime(combined_dataframe["date"])# Convert "date" to datetime type
    combined_dataframe["year"] = combined_dataframe["date"].dt.year# Extract the year

    combined_dataframe_yearly_averages = combined_dataframe.groupby('year').mean()# Calculate the yearly averages of each collumn using a dataframe manipulation function.
    combined_dataframe_yearly_averages = combined_dataframe_yearly_averages.reset_index() #Resets the index of combined_dataframe_yearly_averages 

    #Define the title labels of our new dataframe.
    combined_dataframe_yearly_averages = combined_dataframe_yearly_averages[['year','temperature_2m_mean', 'cloud_cover_mean', 'relative_humidity_2m_mean', 'precipitation_sum']]

    #We save this comvines dataframe to a csv so we are able to call it within all components of the program as it is now a global file.
    combined_dataframe_yearly_averages.to_csv('combined_historicalWeather.csv', sep=',', index=False, encoding='utf-8')
    
    #We can return the dataframe as a single dataframe that contains all the combined historical weather values that have been processed to be in yearly time segments. 
    return combined_dataframe_yearly_averages


# %% [markdown]
# Retrieve Forecasted Weather Data

# %%
def openMeteo_forecasted_climate(latitude, longitude, startDate):
    #We only input the start date as our program will always use the longest possible forecast length automatically. 
    #The longest possible forecast length is 16 days after the current date.

    # Setup the Open-Meteo API client with cache and retry on error.
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    #openMeteo has a maximum forecast 16 days after the start date. This is 15 days as we are including today's date as it is a 0 indexed counting system.

    date_placeholder = datetime.strptime(startDate, '%Y-%m-%d')# Parse the ISO8601 startDate to extract what the day is.
    endDate = date_placeholder + timedelta(days = 15) #Add 15 days to the current date
    endDate = endDate.strftime('%Y-%m-%d')# Format the result back to ISO 8601 and set that as the endDate of the weather forecast.

    #all required weather variables to retrieve are listed here
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
    #The coorindate components
    "latitude": latitude,
	"longitude": longitude,

    #We want data in the highest quality possible so we will retrieve data in hourly steps.
    #Hourly steps means that there will be more datapoints per day however we only have a maximum of 16 days so our system will be able to handle it.
    #Name the four metrics we want predictions for.

	"hourly": ["temperature_2m", "cloud_cover", "relative_humidity_2m", "precipitation", ],

    #Define the start and the end data.
	"start_date": startDate,
	"end_date": endDate
    }

    #Pass the defined parameters into the forecasting openMeteo API.and save the responses.
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models.
    #We process all the responses sequentially so it is clear as to what location is related to what response.
    response = responses[0]

    #We save the identifiers for the program in case errors occur and we need to debug exactly what has happened and where forecasted weather data has been retrieved for.
    debugCoordinates = (f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    debugElevation = (f"Elevation {response.Elevation()} m asl")
    debugTimezone = (f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    debugTimezoneDifference = (f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
    
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()

    #Create an hourly time collumn for the data.
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
        )}
    
    #Save each component that has been retrieved from the forecasted weather API insto a single hourly data dataset.
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation

    hourly_data = pd.DataFrame(data = hourly_data) # Convert dataset into a panda's dataframe

    return hourly_data #Return the hourly dataframe.

def retrieve_forecasted_climate(crop): #This entire function can retrieve all the forecasted weather data for a crop and accordingly merge all its datasets.
    global forecastedWeather1, forecastedWeather2, forecastedWeather3
    #We make these dataframes have a global scope so they can be accessed across the entire program.

    currentDate = datetime.today()#Retrieve the current data in the correct ISO8601 format for the API.
    currentDate = currentDate.strftime('%Y-%m-%d')

    #We use a for loop to loop through all three crop locations and extract the location and the respective coordinates.
    #This process occurs three times as we need to get the 1st, 2nd and 3rd forecasted weather dataset. This is because there are three major locations where each crop is grown.
    for i in range (3):
        latitude, longitude = globals()[crop + 'Coordinates'][i]#Using globals() instead of using multiple if statements for each crop we can write efficient code. This means that we can directly call the crop's coordinates.
        coordinateValidity = verifyCoordinateValidity(latitude, longitude)#We use validation to verify that the coordinates are correct and valid. 

        #If the specific coordinaetes are valid we can generate each forecasted weather dataframe one at a time.
        if coordinateValidity == 'valid values':
            globals()['forecastedWeather' + str(i + 1)] = openMeteo_forecasted_climate(latitude, longitude, currentDate) #Generates 3 different dataframes, one for each country
        elif coordinateValidity == 'invalid values':
            errorAlert('Crop information database contains invalid longitude and latitude coordinates.','hard reset')
        else:
            errorAlert('Coordinates could not be validated. No data has been requested from the API OpenMeteo.','hard reset')

    forecastedWeather1.to_csv('forecastedWeather1.csv', sep=',', index=False, encoding='utf-8') #Saves the dataset as a .csv file. From nupmy to pandas dataframe.
    forecastedWeather2.to_csv('forecastedWeather2.csv', sep=',', index=False, encoding='utf-8') 
    forecastedWeather3.to_csv('forecastedWeather3.csv', sep=',', index=False, encoding='utf-8')

    #Saves the dataset as a .csv file. This allows for physical debugging as we have can see the .csv file in human readable format as a csv is tabular.
    #.csv files and dataframes are interchangeable so it is efficient to save all weather datasets as CSVs. This also means that it is globally accesible across the entire program.
    forecastedWeather1 = pd.read_csv('forecastedWeather1.csv', header=0)
    forecastedWeather2 = pd.read_csv('forecastedWeather2.csv', header=0)
    forecastedWeather3 = pd.read_csv('forecastedWeather3.csv', header=0)


    #This is where we manually multiply all the datasets with the respective percentage volume which that location produces.
    #We multiple every collumn within the dataset by the respective volume spread that location produces.
    # 
    #This is possible due to the consistent indexing and variable naming scheme within our program as the index of the coordinate is equal to the index of the percentage spread.
    #
    forecastedWeather1["temperature_2m"] = (forecastedWeather1["temperature_2m"] * globals()[crop + 'VolumeSpread'][0])
    forecastedWeather1["cloud_cover"] = (forecastedWeather1["cloud_cover"] * globals()[crop + 'VolumeSpread'][0])
    forecastedWeather1["relative_humidity_2m"] = (forecastedWeather1["relative_humidity_2m"] * globals()[crop + 'VolumeSpread'][0])
    forecastedWeather1["precipitation"] = (forecastedWeather1["precipitation"] * globals()[crop + 'VolumeSpread'][0])

    forecastedWeather2["temperature_2m"] = (forecastedWeather2["temperature_2m"] * globals()[crop + 'VolumeSpread'][1])
    forecastedWeather2["cloud_cover"] = (forecastedWeather2["cloud_cover"] * globals()[crop + 'VolumeSpread'][1])
    forecastedWeather2["relative_humidity_2m"] = (forecastedWeather2["relative_humidity_2m"] * globals()[crop + 'VolumeSpread'][1])
    forecastedWeather2["precipitation"] = (forecastedWeather2["precipitation"] * globals()[crop + 'VolumeSpread'][1])

    forecastedWeather3["temperature_2m"] = (forecastedWeather3["temperature_2m"] * globals()[crop + 'VolumeSpread'][2])
    forecastedWeather3["cloud_cover"] = (forecastedWeather3["cloud_cover"] * globals()[crop + 'VolumeSpread'][2])
    forecastedWeather3["relative_humidity_2m"] = (forecastedWeather3["relative_humidity_2m"] * globals()[crop + 'VolumeSpread'][2])
    forecastedWeather3["precipitation"] = (forecastedWeather3["precipitation"] * globals()[crop + 'VolumeSpread'][2])

    #As all the historical weather dataframes have been processed we can just add them using an addition function which adds the respective numbers.
    combined_dataframe = forecastedWeather1 + forecastedWeather2 + forecastedWeather3

    #As given by openMeteo's documentation future forecasts may contain missing data.(empty values)
    #Instead of physically removing datapoints we can just extract complete datapoints and equate that to the original dataset.
    #This effectively deletes empty datapoints as they aren't included within the new dataframe.
    combined_dataframe = combined_dataframe.loc[(combined_dataframe['temperature_2m'] != None) & 
                                                (combined_dataframe['cloud_cover'] != None) & 
                                                (combined_dataframe['relative_humidity_2m'] != None) &
                                                (combined_dataframe['precipitation'] != None)]
    
    combined_dataframe['date'] = pd.to_datetime(combined_dataframe['date'], format='mixed')  # Ensure 'date' is datetime type

    #Defines the split of the data into four time segments. Each of which has a length of 4 days.
    combined_dataframe['time_segment'] = pd.cut(combined_dataframe['date'], bins=4, labels=['Days 1-4', 'Days 5-8', 'Days 9-12', 'Days 13-16'])

    # Calculate mean values for each grouped time segment. This means that we have four different datapoints instead of a single datapoint.
    combined_dataframe = combined_dataframe.groupby('time_segment', observed=False).mean()
    combined_dataframe.to_csv('combined_forecastedWeather.csv', sep=',', index=False, encoding='utf-8')#Save it as a csv so we can use this dataframe for debugging and access it from the rest of the program.

    return combined_dataframe

# %% [markdown]
# Retrieve Current Agricultural Commodity Price

# %%
def get_dataset_link(mainWebsite_url): # Returns the path of commodity price dataset from the World Bank Group Commodity Markets page.
    #url: The URL of the World Bank Commodity Markets page.

    #We use validation here with the implementation of a try: except: statement
    #This means that we can catch the error before it crashes the entire program.

    #Reasons that we can't access the website may be a firewall or a lack of an internet connection so we need to communicate these to our users.
    #an errorAlert within the except clause prints the error why the user can't access the web page and executes an action to fix it.
    try:

        page = requests.get(mainWebsite_url)
        page.raise_for_status() # Verifies the webpage is reachable.
        html = BeautifulSoup(page.text, 'html.parser') # Extracts the html

        # Find the specific link containing "Annual prices"
        xlsx_link = html.find("a", href=True, string="Annual prices")

        if xlsx_link:
            return xlsx_link["href"]
        else :
            errorAlert('Could not access commodity market data','soft reset')

    except requests.exceptions.RequestException as error: # Prevents the code from crashing if the user can't access the webpage and prints the reason.
        errorAlert(f"Error accessing web page>>> {error}","hard reset")

def download_dataset(fileURL): 
    try:#Validation is also used here as we are downloading the entire file chunk by chunk as you can see.
        #If for any reason while downloading it fails the error is caught by the except clause and prints the error.
        response = requests.get(fileURL, stream=True)
        response.raise_for_status()

        with open('commodityPrices.xlsx', 'wb') as f:#Downloads the entire file
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return os.path.abspath('commodityPrices.xlsx')#Returns the absolute location where the data is stored.

    except requests.exceptions.RequestException as error: #Error handling
        errorAlert(f"Error downloading file>>> {error}","hard reset")

def retrieve_commodity_prices(): #Retrieves current and historical agricultural prices.
    commodityDatasetLink = get_dataset_link(worldBank_url)
    commodityPricesPATH = download_dataset(commodityDatasetLink)

    #Our entire program uses dataframes to process data due to openMeteo's dependancy on them and the embedded data manipulation that can be easily used on them.
    commodityPricesDataframe = pd.read_excel(commodityPricesPATH, sheet_name = 'Annual Prices (Nominal)')#We read the retrieved dataset as a pandas dataframe.

    for i in range (5):#We drop 5 rows of empty spaces to make the titles be the clear collumn headers of the datframe.
        commodityPricesDataframe = commodityPricesDataframe.drop(index=(i))

    commodityPricesDataframe = commodityPricesDataframe.iloc[:, [0,11,24,28,35,45]]#We extract only the relevant commodities which we are using to make predictions to be the entire dataset


    commodityPricesDataframe = commodityPricesDataframe.drop(commodityPricesDataframe.index[[1, 2]], axis=0)
    
    return commodityPricesDataframe


# %% [markdown]
# Classify Weather Data

# %%
def classify_commodityPrice_Dataframe(crop,commodityPricesDataframe):#Labels to the selected crop's change in price as either positive or negative.

    #This for loop matches the crop with it's location within the cropIndex array.

    #The index within the crop in the cropIndex array is the same as the index within the commodity prices dataframe.

    for i in range(len(cropIndex)):
        if cropIndex[i] == crop:
            cropRow_DataframeReference = i
            break
    
    commodityPricesDataframe = commodityPricesDataframe.iloc[:, [0,i + 1]]#Extract only the crop collumn you need and the dates.

    pd.set_option('display.max_rows', None)#Display the maximum amount of rows and collumns so all available data is able to be accessed,
    pd.set_option('display.max_columns', None)

    
    commodityPricesDataframe = commodityPricesDataframe.iloc[1:].reset_index(drop=True)
    commodityPricesDataframe.iloc[:, 1] = pd.to_numeric(commodityPricesDataframe.iloc[:, 1])

    #Filters out unnecessary errors produced by the program
    warnings.filterwarnings('ignore', category=FutureWarning, message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated")

    # Calculate percentage change
    commodityPricesDataframe['Percentage Change'] = commodityPricesDataframe.iloc[:, 1].pct_change() * 100
    commodityPricesDataframe['Percentage Change'] = commodityPricesDataframe['Percentage Change'].astype(float).fillna(0)


    # Round percentage change to an integer
    commodityPricesDataframe['Percentage Change'] = commodityPricesDataframe['Percentage Change'].round(0)

    commodityPricesDataframe['Percentage Change'] = (commodityPricesDataframe['Percentage Change'] / 10).round(0) * 10

    # Convert percentage change to correct multiplier format
    commodityPricesDataframe['Percentage Change'] = 1 + (commodityPricesDataframe['Percentage Change'] / 100)

    commodityPricesDataframe['Percentage Change'] = commodityPricesDataframe['Percentage Change'].astype(str)
    return commodityPricesDataframe#Returns the processed dataframe.


# %% [markdown]
# SVM

# %%

def trainingSupportVectorMachine(historicalWeather, historicalPrice, hyperparameters):
    
    X = historicalWeather # The weather is the feature and contains the four climate metrics
    y = historicalPrice # The labelled prices are the labels and is what we are trying to train our SVM to accurately predict.
    
    if hyperparameters == 'no hyperparameters': # If there are no given hyperparameters set it to default ones.
        C_value, gamma_value, kernel_value = 10000, 0.01, 'rbf'
    else:
        C_value, gamma_value, kernel_value = hyperparameters['C'], hyperparameters['gamma'], hyperparameters['kernel']#This allows advanced users to have access to custom hyperparameters when they train their model.
    
    X.drop('year', axis=1, inplace=True) 
    y = y.iloc[:, 2:]
    y=y.values.ravel() #Turns dataframe into a 1 dimensional array as we only have a single collumn we can convert it into a single row.

    if len(X) != len(y):#This involves validation as the model checks if there is missing data. This will happen if one dataset is larger than another dataset so therefore there are missing datapoints.
        errorAlert('missing data. The length of historical weather and price data do not match.','hard reset')

    # Scale the data so it has a mean of 0 and a standard deviation of 1. This gives us previously described performance increases.
    X_scaled = scale(X) 

    # Train the SVM on ALL data
    clf_svm = SVC(C=C_value, gamma=gamma_value, kernel=kernel_value)
    clf_svm.fit(X_scaled, y) 

    return clf_svm#Return the trained SVM model.


# %%
#Outputs the optimal hyperparameters as a dictionary and creates a processing txt file.
def findOptimal_hyperparameters(crop): 
    param_grid = [ 
    {'C':[1, 10, 100, 1000, 10000], #regularazation parameter C.
    'gamma': [ 1, 0.1, 0.01, 0.001],#Different gamme values
    'kernel': ['rbf', 'poly']}]#We can pick different kernel functions ,but RBF is works for most.

    X = retrieve_historical_climate(crop)
    commodityPricesDataframe = retrieve_commodity_prices()
    y = classify_commodityPrice_Dataframe(crop ,commodityPricesDataframe)
    y = y.iloc[:, 2:] 

    y = y.values.ravel()

    X.drop('year', axis=1, inplace=True)

    if len(X) != len(y):
        errorAlert('missing data. The length of historical weather and price data do not match.','hard reset')
    
    X_scaled = scale(X)


    # Run GridSearchCV in a subprocess and capture output. Virtual environment!
    command = ["python3", "-c",
               "from sklearn.model_selection import GridSearchCV; " 
               "from sklearn.svm import SVC; "
               "import numpy as np; "# Import numpy if needed for X_scaled and y
               "param_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000], "
               "'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], "
               "'kernel': ['rbf', 'poly']}; "
               "optimal_params = GridSearchCV(SVC(), param_grid, cv=10, scoring='accuracy', verbose=2); "
               # Assuming X_scaled and y are numpy arrays converted into a list. As subprocess is a virtual environement.
               f"X_scaled = np.array({X_scaled.tolist()}); "  # Convert to list for string representation
               f"y = np.array({y.tolist()}); "
               "optimal_params.fit(X_scaled, y);"
               "print(optimal_params.best_params_)"
              ]
    
    #Run the command above and then dump all the outputs line by line into a text file. 
    with open('processingOutput.txt', 'w') as outfile:
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT, text=True)
    
    with open('processingOutput.txt', 'r') as file:
        for line in file:
            pass
        optimalHyperparameters = line
        optimalHyperparameters = optimalHyperparameters.strip()

        #Extract the optimal hyperparameters from the last line of the outputted text file and return it to the user.
        #We use ast.literal_eval to interpret the string in the text file into a dictionary.
        optimalHyperparameters = ast.literal_eval(optimalHyperparameters)
        
        
    return optimalHyperparameters

# %%
#This is the prediction layer. We take the entire model and the crop and produce a prediction based on forecasted weather data.
def predict(crop,model):

    #We retrieve and standardise the forecasted weather
    futureWeather_dataFrame = retrieve_forecasted_climate(crop)

    #Delete the first date collumn as it is unnecessary and not a numerical value.
    futureWeather_dataFrame = futureWeather_dataFrame.drop(futureWeather_dataFrame.columns[0], axis=1)
    

    #Standardise and scale the data.
    futureWeather_dataFrame_scaled = scale(futureWeather_dataFrame)

    #We input the forecated weather data into the model producing the predictions which we return as an array.
    prediction = model.predict(futureWeather_dataFrame_scaled)

    # Convert to array of floating-point numbers
    prediction = prediction.astype(np.float64)

    yearlyChange = np.array(prediction)
    fourDayChange = 1 + (((yearlyChange - 1) / 365) * 4)

    prediction = fourDayChange

    return prediction 

# %%
def impute_anomalies(dataframe):

    #This is the k value and defines how large the bounds will be.
    #A larger k value will mean the bound will cover less of the entire dataset.
    #This means that the function will be less sensitive to anomalies.
    k = 2

    for column in dataframe.columns:  

        #This checks if the column only contains numerical numbers as we can't impute string values.
        if pd.api.types.is_numeric_dtype(dataframe[column]):
           
            #Find the upper and the lower quartiles.
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)

            IQR = Q3 - Q1#Find the interquartile range

            #Define the upper and lower bounds which data have to be within.
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            #.loc is used to access and modify specific elements within the specific collumn.
            #This means there is no need to create an inefficient for loop to loop through all the values and columns.

            #The condition selects only elements that fall outside of the bounds we defined.
            #These specific elements are then replaced with the mean of the collumn.
            dataframe.loc[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound), column] = dataframe[column].mean()


    return dataframe
    
def impute_missing(dataframe):

    for column in dataframe.columns:  
          #This checks if the column only contains numerical numbers as we can't impute string values.
          if pd.api.types.is_numeric_dtype(dataframe[column]):

            #.loc is used to access and modify specific elements within the specific collumn.
            #This means there is no need to create an inefficient for loop to loop through all the values and columns.

            #These specific elements are then replaced with the mean of the collumn.
            dataframe[column] = dataframe[column].astype(float).fillna(dataframe[column].mean())
    
    return dataframe
# %%
custom_cValue = None
custom_gammaValue = None

def customHyperparameters(c,gamma):
    global custom_cValue
    global custom_gammaValue
    custom_cValue = c
    custom_gammaValue = gamma

# %%
def initialiseParameters():
    custom_cValue = None
    custom_gammaValue = None

# %% [markdown]
# Graphical User Interface

# %%
def clear_page():
    #Clears the entire page of widgets. So the entire page is empty.
    for widget in root.winfo_children():
        widget.destroy()

def crop_selection_page():
    #We open the crop selection page as the starting page for the user to pick from a selection of crops.
    clear_page()

    # Header
    header_frame = tk.Frame(root, bg="#80ff97")
    header_frame.pack(fill="x")
    tk.Label(header_frame, text="SELECT A CROP BELOW", font=("Arial", 24, "bold")).pack(pady=10)

    # Information Box
    info_frame = tk.LabelFrame(root, text="Information", bd=2, relief="solid")
    info_frame.pack(fill="x", padx=10, pady=10)
    info_text = "This box provides static information. For example, the tutorial prompts below.\n\n"
    info_text += "Select a crop to analyze.\n"
    info_text += "Access advanced settings to edit hyperparameters and other settings.\n"
    tk.Label(info_frame, text=info_text, justify="left", wraplength=500).pack(padx=10, pady=10)

    # Buttons Frame
    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", padx=10, pady=10)

    # Advanced Settings Button
    advanced_button = tk.Button(button_frame, text="Advanced\nSettings", command=lambda :advanced_settings(), fg="white", bg="blue",
                                font=("Arial", 14, "bold"), width=10)
    advanced_button.pack(side="left", padx=5, pady=5)


    # Crop Buttons
    for crop in ("cocoa", "wheat", "corn", "soybean", "sugar"):
        ttk.Button(button_frame, text=crop, command=lambda c=crop: processing_page(c), width=10).pack(side="left", padx=10, pady=50)

def processing_page(crop):
    # This is the processing page where all the data is being processed in the background. 
    # The user will get updates as to the progress and if any errors occur during the entire process.
    clear_page()

    # Header of the processing page...
    header_frame = tk.Frame(root, bg="#80ff97")
    header_frame.pack(fill="x")
    tk.Label(header_frame, text="PROCESSING PAGE", font=("Arial", 24, "bold"), bg="#80ff97").pack(pady=10)

    # Progress Bar that shows the progress of the entire solution.
    progress_frame = tk.Frame(root)
    progress_frame.pack(fill="x", padx=10, pady=10)
    progress_canvas = tk.Canvas(progress_frame, height=20, bg="lightgray", highlightthickness=0)
    progress_canvas.pack(fill="x")
    progress_bar = progress_canvas.create_rectangle(0, 0, 0, 20, fill="#80ff97")#The rectangle represents the progress bar we can change the rectangle's width to represent the change in progress over time.

    # makes the labelled error and progress frames that are side by side to eachother.
    error_progress_frame = tk.Frame(root)
    error_progress_frame.pack(fill="x", padx=10, pady=10)
    error_frame = tk.LabelFrame(error_progress_frame, text="Errors:", bd=2, relief="solid", bg="#ffbfad") 
    error_frame.pack(side="left", fill="both", expand=True, padx=5)
    progress_info_frame = tk.LabelFrame(error_progress_frame, text="Progress", bd=2, relief="solid", bg="#80ff97") 
    progress_info_frame.pack(side="right", fill="both", expand=True, padx=5)

    # Error Block
    error_label = tk.Label(error_frame, text="None", bg="#ffbfad") 
    error_label.pack(padx=10, pady=10)

    # Progress Block
    progress_label = tk.Label(progress_info_frame, text="", bg="#80ff97")  # Empty label initially
    progress_label.pack(padx=10, pady=10)

    # We can use this function to update the text within the progress block
    def update_progress_text(text):
        progress_label.config(text=text)
    
    #We can use this function to update the text within the error block.
    def update_error_text(text):
        error_label.config(text=text)

    # Function to update the progress on the progress bar this makes the bar larger.
    def update_progress(percent):
        progress_canvas.coords(progress_bar, 0, 0, progress_canvas.winfo_width() * (percent / 100), 20)
    
    def processingTasks():

        try:#Validation has occured as we are making sure that historical weather and price data has been retrieved succesfully/
            historicalWeather = retrieve_historical_climate(crop)
            commodityPrices = retrieve_commodity_prices()

            #We update all the progress indicators to show this.
            update_progress_text('Retrieved Weather + Price Data')
            update_progress(10)
            
            progress_canvas.update_idletasks()#This prevents tkinter from running anything past this point before the previous tasks have been completed / rendered.



        except:
            update_error_text('ERROR - Retrieving Data see the python terminal')#We catch the errors and display it to the user both in tkinter and in the python command line interface.
            errorAlert('Unfortunately we were unable to retrieve either historical weather or commodity prices from the internet. Check your internet connection or firewall','soft reset')


        try:#We validate whether the commodity prices have been succesfully labelled.
            labelledPrices = classify_commodityPrice_Dataframe(crop, commodityPrices)#Label the commodity prices

            #Update the progress to show this
            update_progress(20)
            update_progress_text('Labelled Crop Price Data')


            progress_canvas.update_idletasks()
        except:
            update_error_text('ERROR Labelling commodity prices')#We catch the errors and display it to the user both in tkinter and in the python command line interface.
            errorAlert('Unfortunately the commodity prices were not able to be either accessed or labelled.','soft reset')

        update_progress(30)
        update_progress_text('Training the model...')
        findOptimal_hyperparameters(crop)


        #If the user has not set custom C and Gamma hyperparameter values then we can use the grid-search cross validation to find these optimal values and deliver it to the user.

        if custom_cValue == None and custom_gammaValue == None:
            trainedModel = trainingSupportVectorMachine(historicalWeather,labelledPrices, 'no hyperparameters') #The program will optimise it itself with custom hyperparameters.
        else:
            #If the user has set custom C and Gamma hyperparameter values then we will invoke this else statement. Within this we will use these values and pass it within a dictionary.
            custom_hyperparameterDictionary = {
                'C' : custom_cValue,
                'gamma' : custom_gammaValue,
                'kernel': 'rbf'
                }
            
            #We can then pass these values in the dictionary into training the model.
            trainedModel = trainingSupportVectorMachine(historicalWeather, labelledPrices, custom_hyperparameterDictionary)

        update_progress_text('Predicting the future...')
        update_progress(80)

        
        #We want the predictions to have a global scope so we can access it across the entire program.
        global predictions

        predictions = [] #We set the predictions to make sure it is empty before we add the new predictions to the dataset.

        predictions = predict(crop, trainedModel)

        update_progress_text('Complete')
        update_progress(100)

    
    root.after(0, processingTasks())#We run the processingTasks function 100ms or 0.1s after the entire program to let tkinter render all their content correctly.
    output_page(crop)

def output_page(crop):#This is the output page where we output the graph and the predictions onto a suitable interactive page.
    
    clear_page()
    import datetime
    
    commodity_prices_df = classify_commodityPrice_Dataframe(crop, retrieve_commodity_prices())

    global predictions
    predictions = predictions.astype(np.float64)#Produces a numerical array of all the predictions from a string to a float datatype.

    # Creates the plot
    fig, ax = plt.subplots() 
    ax.plot(commodity_prices_df.iloc[:, 0], commodity_prices_df.iloc[:, 1])  # Plotting of the year and the price...
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")

    ax.set_title(f"{crop.capitalize()} Prices Over Time") #Title of the graph (crop name is here) Prices Over Time

    # Retrieve all the dates we need...
    today = datetime.date.today()
    current_year = today.year

    # Get the last datapoints from the dataset.
    last_year = int(commodity_prices_df.iloc[-1, 0])  
    last_price = commodity_prices_df.iloc[-1, 1]

    # Plot today's data (assuming the price is the same as the last year's price) We are assuming this dataset is the most up-to-date dataset we have access too.
    ax.plot(current_year, last_price, 'ro', label="Today's Price")

    # Draw a line connecting the last point and today's point
    ax.plot([last_year, current_year], [last_price, last_price], 'r--', label="Connecting Line")  # Draw a line between the points


    last_date = datetime.date.today()  # We set the start to be todays date.
    last_price = commodity_prices_df.iloc[-1, 1]  # Retrieve the last price

    for i in range(min(4, len(predictions))):  # Plot up to 4 predictions 

        next_date = last_date + datetime.timedelta(days=4)#Add four days to get to the next point.
        next_price = last_price * predictions[i]  # Calculate the next price using the predictions array.

        # Determine line color based on price change green is positive red is negative change.
        if next_price > last_price:
            line_color = 'g-'
        else:
            line_color = 'r-'

        # Convert dates to years for plotting
        last_year = last_date.year + (last_date.timetuple().tm_yday / 365.25)  # Accounts for leap years as 1/4 = 0.25 
        next_year = next_date.year + (next_date.timetuple().tm_yday / 365.25) #We need to do this as our entire dataset has a yearly axes and we can't plot months and days.

        ax.plot([last_year, next_year], [last_price, next_price], line_color)#Plot the connecting line with the correct line colour.

        last_date = next_date
        last_price = next_price
    
    def zoom_to_predicted_prices():#Allow time scale viewing to see specific segments of the graph.
        
        start_date = datetime.date.today() #We set this to today's current date beforehand so we can use this same variable.
        
        dates = []
        prices = []
        
        last_price = commodity_prices_df.iloc[-1, 1]  # Get the last price from the dataframe using the iloc dataframe function.
        
        current_date = start_date  # Initialize current_date with start_date

        for i in range(len(predictions)):  # Iterate through the predictions
            dates.append(current_date)  # Add the current date to the dates list
            prices.append(last_price * predictions[i])  # Calculate and add the predicted price

            current_date = current_date + datetime.timedelta(days=4)  # Increment the current_date by 4 days

        years = [d.year + (d.timetuple().tm_yday / 365.25) for d in dates]# Convert dates to years for plotting as our x axis is in years.
        
        # Set x,y zoom
        ax.set_xlim(min(years) - 0.15, max(years) + 0.15)  # Zoom to see x axis years
        ax.set_ylim(min(prices) * 0.99, max(prices) * 1.01)  # Zoom to see y axis predicted prices

        canvas.draw()  # draw the canvas again to reflect the zoom.

    zoom_button = ttk.Button(root, text="Zoom In to see Current Changes", command=zoom_to_predicted_prices)#Button to zoom in
    zoom_button.pack(pady=10)

    ax.legend()

    # Embeds the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Adds the matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # Creates and packs the back button
    back_button = ttk.Button(
        root,
        text="Back to Crop Selection",
        command=lambda: [crop_selection_page()],
    )
    back_button.pack(pady=50)

#This will be the advanced settings page where users are able to customise the hyperparameters.
def advanced_settings():
    clear_page()

    # Information Box
    info_frame = tk.LabelFrame(root, text="Information", bd=2, relief="solid")
    info_frame.pack(fill="x", padx=10, pady=10)
    info_text = "This is the advanced settings page. Here you can adjust the hyperparameters 'Gamma' and 'C'.\n\n"
    info_text += "Please enter numerical values for both hyperparameters."
    tk.Label(info_frame, text=info_text, justify="left", wraplength=500).pack(padx=10, pady=10)

    # Input Frame
    input_frame = tk.Frame(root)
    input_frame.pack(pady=20)

    # Gamma Value
    tk.Label(input_frame, text="Gamma Value:").grid(row=0, column=0, sticky="e")
    gamma_entry = tk.Entry(input_frame)
    gamma_entry.grid(row=0, column=1, padx=5)

    # C Value
    tk.Label(input_frame, text="C Value:").grid(row=1, column=0, sticky="e")
    c_entry = tk.Entry(input_frame)
    c_entry.grid(row=1, column=1, padx=5)

    def submit_values():
        try:
            gamma = float(gamma_entry.get())
            c = float(c_entry.get())

            customHyperparameters(c,gamma)
            crop_selection_page()
        except ValueError:#Here we use validation to make sure the user inputs integer values for both gamma and c.
            messagebox.showerror("Error", "Please enter valid numerical values for both Gamma and C.")

    # Submit Button
    submit_button = ttk.Button(root, text="Submit", command=submit_values)
    submit_button.pack(pady=10)

if __name__ == "__main__":
    initialiseParameters()
    root = tk.Tk()
    root.title("scion")
    crop_selection_page()

    root.mainloop()
