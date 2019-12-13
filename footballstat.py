import pandas as pd
import numpy as np

#read csv data
xg_data = pd.read_csv('epl_xg.csv')

# add new goal difference columns
xg_data['GD'] = xg_data['G'] - xg_data['GA']
xg_data['xGD'] = xg_data['xG'] - xg_data['xGA']
xg_data['NPxGD'] = xg_data['NPxG'] - xg_data['NPxGA']

# plot real vs expected
xg_data['GD_vs_xGD'] = xg_data['GD'] - xg_data['xGD']
xg_data = xg_data.sort_values(by=['GD_vs_xGD'], ascending=False)

cols = ['G', 'GA', 'xG', 'xGA', 'NPxG', 'NPxGA', 'GD', 'xGD', 'NPxGD']
for col in cols:
    xg_data['%s_pg' % col] = xg_data[col] / xg_data['Games']

#load next fixture
fixtures = pd.read_csv('epl_fixtures.csv')

xg_data_pg = xg_data[['Team', 'xG_pg', 'xGA_pg']]
fixtures = pd.merge(fixtures, xg_data_pg, left_on='Home_Team', right_on='Team')
fixtures = pd.merge(fixtures, xg_data_pg, left_on='Away_Team', right_on='Team')
fixtures = fixtures.drop(['Team_x', 'Team_y'], axis=1)
fixtures = fixtures.rename(columns={
    'xG_pg_x': 'xG_pg_home', 'xGA_pg_x': 'xGA_pg_home', 'xG_pg_y': 'xG_pg_away', 'xGA_pg_y': 'xGA_pg_away'})

#calculated adjusted stat
xG_avg = xg_data_pg['xG_pg'].mean()
matchups = fixtures.copy()
matchups['xG_adjusted_home'] = (matchups['xG_pg_home'] * matchups['xGA_pg_away']) / xG_avg
matchups['xG_adjusted_away'] = (matchups['xG_pg_away'] * matchups['xGA_pg_home']) / xG_avg

#calculate home advantage
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
