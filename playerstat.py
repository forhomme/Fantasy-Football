import pandas as pd
import re
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import datetime

injuries_url = 'https://www.fantasyfootballscout.co.uk/fantasy-football-injuries/'
injury_tables = pd.read_html(injuries_url, encoding='utf-8')
injuries = injury_tables[0]

# formating name
injuries['first_name'] = injuries['Name'].str.extract(r'\(([\w-]+)\)', expand=True)
injuries['last_name'] = injuries['Name'].str.extract(r'^(.+?)\s\(', expand=True)
injuries['full_name'] = injuries['first_name'] + ' ' + injuries['last_name']

# remove NaN value
injuries['last_name'] = injuries['last_name'].str.normalize('NFKD').str.encode(
    'ascii', errors='ignore').str.decode('utf-8')
injuries['full_name'] = injuries['full_name'].fillna(injuries['Name'])
injuries['full_name'] = injuries['full_name'].fillna(injuries['last_name'])
injuries = injuries[['full_name', 'Club', 'Status', 'Return Date', 'Latest News', 'Last Updated']]
injuries.columns = injuries.columns.str.lower().str.replace(' ', '_')

# convert last_updated to datetime
pd.set_option('mode.chained_assignment', None)
injuries['last_updated'] = pd.to_datetime(injuries['last_updated'], format='%d/%m/%Y')
print(injuries)
