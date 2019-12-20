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
xg_url = 'https://understat.com/league/EPL'
# xg_data = requests.get(xg_url)
# xg_html = xg_data.content

# selenium to access data
options = webdriver.FirefoxOptions()
options.add_argument('headless')
driver = webdriver.Firefox(firefox_options=options)
driver.get(xg_url)
soup = BeautifulSoup(driver.page_source, 'lxml')

# access element from inspect element
headers = soup.find('div', attrs={
    'class': 'chemp margin-top jTable'}).find('table').find_all('th', attrs={'class': 'sort'})

# iterate over header
headers_list = []
for header in headers:
    headers_list.append(header.get_text(strip=True))

# call elemet in the header
body = soup.find('div', attrs={'class': 'chemp margin-top jTable'}).table.tbody
all_row_list = []
for tr in body.find_all('tr'):
    row = tr.find_all('td')
    current_row = []
    for item in row:
        current_row.append(item.get_text(strip=True))
    all_row_list.append(current_row)

xg_df = pd.DataFrame(all_row_list, columns=headers_list)
xg_df = xg_df.replace('Wolverhampton Wanderers', 'Wolverhampton')

driver.close()

# get rid unwanted text
xg_df['xG'] = xg_df['xG'].str.extract(r'^(.+?)[\+\-]', expand=True)
xg_df['xGA'] = xg_df['xGA'].str.extract(r'^(.+?)[\+\-]', expand=True)
xg_df['xPTS'] = xg_df['xPTS'].str.extract(r'^(.+?)[\+\-]', expand=True)
xg_df['xG'] = xg_df['xG'].fillna(xg_df['G'])
xg_df['xGA'] = xg_df['xGA'].fillna(xg_df['GA'])
xg_df['xPTS'] = xg_df['xPTS'].fillna(xg_df['PTS'])
xg_df = xg_df[['Team', 'M', 'W', 'L', 'G', 'GA', 'PTS', 'xG', 'xGA', 'xPTS']]
print(xg_df)
# display.stop()
