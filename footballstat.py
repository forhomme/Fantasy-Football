import pandas as pd
import numpy as np
import re
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import datetime
from xvfbwrapper import Xvfb
from matplotlib import pyplot as plt
import seaborn as sns

# start xvfb
display = Xvfb()
display.start()

# get from url
xg_url = 'https://understat.com/league/EPL'
# xg_req = requests.get(xg_url)
# xg_html = xg_req.content

# selenium to access data
options = webdriver.FirefoxOptions()
options.add_argument('headless')
driver = webdriver.Firefox(firefox_options=options)
driver.implicitly_wait(10)
driver.get(xg_url)
soup = BeautifulSoup(driver.page_source, 'lxml')

# access element from inspect element
headers = soup.find('div', attrs={'class': 'chemp margin-top jTable'}).find('table').find_all('th',attrs={'class':'sort'})

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

# get rid unwanted text
xg_df['xG'] = xg_df['xG'].str.extract(r'^(.+?)[\+\-]', expand=True)
xg_df['xGA'] = xg_df['xGA'].str.extract(r'^(.+?)[\+\-]', expand=True)
xg_df['xPTS'] = xg_df['xPTS'].str.extract(r'^(.+?)[\+\-]', expand=True)
xg_data = xg_df[['Team', 'M', 'W', 'L', 'G', 'GA', 'PTS', 'xG', 'xGA', 'xPTS']]
xg_data = xg_data.replace('Wolverhampton Wanderers', 'Wolverhampton')
driver.close()

# add new goal difference columns
pd.set_option('mode.chained_assignment', None)
cols = ['M', 'W', 'L', 'G', 'GA', 'PTS', 'xG', 'xGA', 'xPTS']
for col in cols:
    xg_data[col] = pd.to_numeric(xg_data[col])
xg_data['GD'] = xg_data['G'] - xg_data['GA']
xg_data['xGD'] = xg_data['xG'] - xg_data['xGA']

# plot real vs expected
xg_data['GD_vs_xGD'] = xg_data['GD'] - xg_data['xGD']
xg_data = xg_data.sort_values(by=['GD_vs_xGD'], ascending=False)

# Set the plot style and colour palette to use (remember dodgy spelling if you're from the UK!)
sns.set(style='whitegrid')
sns.set_color_codes('muted')

# Initialize the matplotlib figure (f) and axes (ax), and set width and height of the plot
f, ax = plt.subplots(figsize=(12, 10))

# Create the plot, choosing the variables for each axis, the data source and the colour (b = blue)
sns.barplot(x='GD_vs_xGD', y='Team', data=xg_data, color='b')

# Rename the axes, setting y axis label to be blank
ax.set(ylabel='', xlabel='Goal Difference vs Expected Goal Difference')

# Remove the borders from the plot
sns.despine(left=True, bottom=True)
plt.show()

cols = ['G', 'GA', 'xG', 'xGA', 'GD', 'xGD']
for col in cols:
    xg_data['%s_pg' % col] = xg_data[col] / xg_data['M']

# load next fixture
# start xvfb
# display = Xvfb()
# display.start()

# get from url
fixture_url = 'https://us.soccerway.com/national/england/premier-league/20192020/regular-season/r53145/'
# fixture_req = requests.get(fixture_url)
# fixture_html = fixture_req.content

# selenium to access data
# options = webdriver.FirefoxOptions()
# options.add_argument('headless')
# driver = webdriver.Firefox(firefox_options=options)
driver.implicitly_wait(15)
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
fixture_df = fixture_df.rename(columns={'Home team': 'Home_Team', 'Away team': 'Away_Team'})
fixture_df = fixture_df[['Date', 'Home_Team', 'Away_Team']]
fixture_df = fixture_df.replace(['Newcastle United', 'AFC Bournemouth', 'Leicester City', 'Norwich City', 'West Ham United',
                       'Tottenham Hotspur', 'Wolverhampton …','Brighton & Hov…'],
                      ['Newcastle', 'Bournemouth', 'Leicester', 'Norwich', 'West Ham', 'Tottenham', 'Wolverhampton', 'Brighton'])
driver.close()
display.stop()

# calculate over fixture
xg_data_pg = xg_data[['Team', 'xG_pg', 'xGA_pg']]
fixture_df = pd.merge(fixture_df, xg_data_pg, left_on='Home_Team', right_on='Team')
fixture_df = pd.merge(fixture_df, xg_data_pg, left_on='Away_Team', right_on='Team')
fixture_df = fixture_df.drop(['Team_x', 'Team_y'], axis=1)
fixture_df = fixture_df.rename(columns={
    'xG_pg_x': 'xG_pg_home', 'xGA_pg_x': 'xGA_pg_home', 'xG_pg_y': 'xG_pg_away', 'xGA_pg_y': 'xGA_pg_away'})

# calculated adjusted stat
xG_avg = xg_data_pg['xG_pg'].mean()
matchups = fixture_df.copy()
matchups['xG_adjusted_home'] = (matchups['xG_pg_home'] * matchups['xGA_pg_away']) / xG_avg
matchups['xG_adjusted_away'] = (matchups['xG_pg_away'] * matchups['xGA_pg_home']) / xG_avg

# calculate home advantage
matchups['xG_adjusted_home'] = matchups['xG_adjusted_home'] * 1.1
matchups['xG_adjusted_away'] = matchups['xG_adjusted_away'] * 0.87


def win_cs(df, home_goals_col, away_goals_col, n=10000):
    home_win_list = []
    away_win_list = []
    home_cs_list = []
    away_cs_list = []

    for i in range(len(df)):
        avg_home_goals = df.iloc[i][home_goals_col]
        avg_away_goals = df.iloc[i][away_goals_col]

        # simulate random poisson disributin n times
        home_goals_sim = np.random.poisson(avg_home_goals, n)
        away_goals_sim = np.random.poisson(avg_away_goals, n)
        sim = pd.DataFrame({'home_score': home_goals_sim, 'away_score': away_goals_sim})
        sim['home_win'] = np.where(sim['home_score'] > sim['away_score'], 1, 0)
        sim['away_win'] = np.where(sim['away_score'] > sim['home_score'], 1, 0)
        sim['home_clean_sheet'] = np.where(sim['away_score'] == 0, 1, 0)
        sim['away_clean_sheet'] = np.where(sim['home_score'] == 0, 1, 0)
        home_win_list.append(sim['home_win'].sum() / n)
        away_win_list.append(sim['away_win'].sum() / n)
        home_cs_list.append(sim['home_clean_sheet'].sum() / n)
        away_cs_list.append(sim['away_clean_sheet'].sum() / n)

    df['home_win'] = np.asarray(home_win_list)
    df['away_win'] = np.asarray(away_win_list)
    df['home_clean_sheet'] = np.asarray(home_cs_list)
    df['away_clean_sheet'] = np.asarray(away_cs_list)
    return df


matchups = win_cs(df=matchups, home_goals_col='xG_adjusted_home', away_goals_col='xG_adjusted_away')
displaycols = ['Home_Team', 'xG_adjusted_home', 'home_win', 'home_clean_sheet',
               'Away_Team', 'xG_adjusted_away', 'away_win', 'away_clean_sheet']
print(matchups[displaycols])
