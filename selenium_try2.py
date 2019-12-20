import pandas as pd
import re
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import datetime
from xvfbwrapper import Xvfb

# start xvfb
# display = Xvfb()
# display.start()

# get from url
fixture_url = 'https://us.soccerway.com/national/england/premier-league/20192020/regular-season/r53145/'
# fixture_req = requests.get(fixture_url)
# fixture_html = fixture_req.content

# selenium to access data
options = webdriver.FirefoxOptions()
options.add_argument('headless')
driver = webdriver.Firefox(firefox_options=options)
driver.get(fixture_url)
soup = BeautifulSoup(driver.page_source, 'lxml')

# access element from inspect element
headers = soup.find('div', attrs={
    'class': 'table-container'}).find('table').find_all('th')

# iterate over header
headers_list = []
for header in headers:
    headers_list.append(header.get_text(strip=True))

if len(headers_list) < 7:
    headers_list.append('')

# call element in the header
body = soup.find('div', attrs={'class': 'table-container'}).table.tbody
all_row_list = []
for tr in body.find_all('tr'):
    row = tr.find_all('td')
    current_row = []
    for item in row:
        current_row.append(item.get_text(strip=True))
    all_row_list.append(current_row)

fixture_df = pd.DataFrame(all_row_list, columns=headers_list)
driver.close()

fixture_df = fixture_df[['Date', 'Home team', 'Away team']]
fixture_df = fixture_df.replace(['Newcastle United', 'AFC Bournemouth', 'Leicester City', 'Norwich City', 'West Ham United',
                       'Tottenham Hotspur', 'Wolverhampton …','Brighton & Hov…'],
                      ['Newcastle', 'Bournemouth', 'Leicester', 'Norwich', 'West Ham', 'Tottenham', 'Wolverhampton', 'Brighton'])
print(fixture_df)
