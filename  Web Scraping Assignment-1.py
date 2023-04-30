#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[16]:


get_ipython().system('pip install bs4')
get_ipython().system('pip install requests')


# In[4]:


from bs4 import BeautifulSoup
import requests


# In[6]:


#1) Write a python program to display all the header tags from wikipedia.org and make data frame.

webPage = requests.get('https://en.wikipedia.org/wiki/Main_Page')

webPage


# In[8]:


bSoup = BeautifulSoup(webPage.content)

bSoup


# In[13]:


#Write a python program to display all the header tags from wikipedia.org and make data frame.

page_headers = bSoup.find_all(['h1','h2','h3','h4','h5','h6'])


# In[14]:


page_headers


# In[21]:


df = pd.DataFrame(data =page_headers)


# In[22]:


df


# In[23]:


# 2) Write a python program to display IMDB’s Top rated 50 movies’ data (i.e. name, rating, year of release)
# and make data frame.

webPage = requests.get('https://www.imdb.com/list/ls055386972/')

webPage


# In[146]:


bSoup = BeautifulSoup(webPage.content)

bSoup


# In[169]:


f_titles = [] 


# In[170]:


for i in bSoup.find_all('h3', class_='lister-item-header'):
     
        f_titles.append(i.text.replace(".\n", "#").replace("\n", "").replace(")","").replace("(","#").split("#"))
    
f_titles


# In[171]:


column_headings = ['serial','Title','Year','None']

df = pd.DataFrame(data = f_titles ,columns = column_headings)


df


# In[172]:


df.iloc[13].Year = '2008'


# In[173]:


df.iloc[13].Title = 'Taken I'


# In[174]:


df


# In[175]:


df.drop('None', axis='columns', inplace=True)


# In[176]:


df.drop('serial', axis='columns', inplace=True)


# In[177]:


df


# In[184]:


f_rating = []

for i in bSoup.find_all('div', class_='ipl-rating-star small'):
     
        f_rating.append(i.text.replace("\n", "").replace(",",""))
    
f_rating   


# In[186]:


rate_df = pd.DataFrame(data = f_rating, columns =['Rating'])

rate_df


# In[192]:


# df.insert(loc=2, column= 'Rating', value=[i for i in rate_df])

df['Rating'] = rate_df


# In[193]:


df


# In[194]:


IMDB_Top_rated_50_movies = df


# In[195]:


IMDB_Top_rated_50_movies


# In[13]:


# 3) Write a python program to display IMDB’s Top rated 50 Indian movies’ data (i.e. name, rating, year of
# release) and make data frame.

webPage = requests.get('https://www.imdb.com/list/ls077629380/')

webPage


# In[14]:


bSoup = BeautifulSoup(webPage.content)

bSoup


# In[15]:


film_list = []

for i in bSoup.find_all('h3', class_='lister-item-header'):
    film_list.append(i.text.replace(".\n", "#").replace("\n", "").replace(")","").replace("(","#").split("#"))
    
film_list


# In[16]:


df = pd.DataFrame(data=film_list, columns=['Serial','Film_Title','Year_of_Release', 'None'])

df


# In[17]:


df.iloc[29].Year_of_Release = '2013'


# In[18]:


df.iloc[29].Film_Title = 'Lootera I'


# In[19]:


df.iloc[42].Year_of_Release = '2013'


# In[20]:


df.iloc[29].Film_Title = 'D-Day I'


# In[21]:


df.drop(['Serial','None'], axis ='columns', inplace=True)


# In[23]:


df


# In[24]:


df.iloc[9].Year_of_Release = '2013'


# In[29]:


f_rating = []

for i in bSoup.find_all('div', class_='ipl-rating-star small'):
     
        f_rating.append(i.text.replace("\n", "").replace(",",""))
    
f_rating  


# In[30]:


df_rate = pd.DataFrame(data = f_rating, columns=['Rating'])


# In[32]:


df['Rating'] = df_rate

df


# In[33]:


df['Rating'] = df_rate

df


# In[290]:


IMDB_Top_rated_50_Indian_movies = df


# In[291]:


IMDB_Top_rated_50_Indian_movies


# In[361]:


# 4) Write s python program to display list of respected former presidents of India(i.e. Name , Term of office)
# from https://presidentofindia.nic.in/former-presidents.htm and make data frame.

webPage = requests.get('https://presidentofindia.nic.in/former-presidents.htm')

webPage


# In[362]:


bSoup = BeautifulSoup(webPage.content)

bSoup


# In[363]:


president_list = []

for i in bSoup.find_all('div', class_='presidentListing'):
    president_list.append(i.text.replace(".\nhttp", "#").replace(")\nTerm of Office:", "#").replace("(","#").replace("\n","").split("#"))
    
president_list


# In[339]:


pre_df = pd.DataFrame(data=president_list, columns=['President_Name','Birth','Term'])

pre_df


# In[341]:


pre_df.drop('Birth', axis ='columns', inplace=True)


# In[342]:


pre_df


# In[343]:


pre_df.iloc[13].Term = '26 January, 1950 to 13 May, 1962'


# In[349]:


pre_df.iloc[0].Term = '25 July, 2017 to 25 July, 2022'


# In[350]:


pre_df.iloc[1].Term = '25 July, 2012 to 25 July, 2017'


# In[351]:


pre_df.iloc[2].Term = '25 July, 2007 to 25 July, 2012'


# In[352]:


pre_df.iloc[3].Term = '25 July, 2002 to 25 July, 2007'


# In[353]:


pre_df


# In[356]:


Indian_Respected_Former_Presidents = pre_df


# In[357]:


Indian_Respected_Former_Presidents


# In[ ]:


# 5) Write a python program to scrape cricket rankings from icc-cricket.com. You have to scrape and make data frame-
# a) Top 10 ODI teams in men’s cricket along with the records for matches, points and rating.
# b) Top 10 ODI Batsmen along with the records of their team andrating.
# c) Top 10 ODI bowlers along with the records of their team andrating


# In[433]:


# 5a) Top 10 ODI teams in men’s cricket along with the records for matches, points and rating. table-head

webPage = requests.get('https://www.icc-cricket.com/rankings/mens/team-rankings/odi')

webPage

bSoup = BeautifulSoup(webPage.content)

bSoup


# In[437]:


Men_Cricket_List = []

for i in bSoup.find_all('tr', class_='rankings-block__banner'):
    Men_Cricket_List.append(i.text.replace("\n\n\n", "#").replace("\n\n","#").replace("\n","#").split("#"))
    
Men_Cricket_List


# In[438]:


for i in bSoup.find_all('tr', class_='table-body', limit=9):
   
        Men_Cricket_List.append(i.text.replace("\n\n\n", "#").replace("\n\n","#").replace("\n","#").split("#"))       
        
Men_Cricket_List


# In[440]:


MCL_df = pd.DataFrame(data=Men_Cricket_List, columns =['None','Serial','Teams','Team_Abr','Matches','Point','Rating','None','None'])

MCL_df


# In[443]:


MCL_df.drop('None',axis='columns', inplace=True)


# In[444]:


Top_10_ODI_teams_in_men_cricket = MCL_df

Top_10_ODI_teams_in_men_cricket


# In[485]:


# 5b) Top 10 ODI Batsmen along with the records of their team andrating.

webPage = requests.get('https://www.icc-cricket.com/rankings/mens/player-rankings/odi/batting')

webPage

bSoup = BeautifulSoup(webPage.content)

bSoup


# In[486]:


Men_Batmen_Rating = [] 

for i in bSoup.find_all('tr', class_='rankings-block__banner'):
    Men_Batmen_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n\n\n\n\n\n\n\n","#").replace("\n\n\n\n\n\n\n", "#").replace("\n\n\n\n\n","#").replace("n\n\n\n\n\n","").replace("\n\n\n", "#").replace("\n", "").strip().split("#"))
    
Men_Batmen_Rating


# In[487]:



for i in bSoup.find_all('tr', class_='table-body', limit = 9):
    Men_Batmen_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n","#").replace("\n\n\n\n", "#").replace("\n\n","#").replace("\n", "").strip().split("#"))
    
Men_Batmen_Rating


# In[488]:


Top_Ten_Batmen = pd.DataFrame(data=Men_Batmen_Rating, columns=['None','Serial','Player_Name','Team','Rating','Best_Rate','None'])

Top_Ten_Batmen


# In[489]:


Top_Ten_Batmen.drop('None', axis='columns',inplace=True)


# In[490]:


Top_Ten_Batmen


# In[497]:


5# c) Top 10 ODI bowlers along with the records of their team andrating

webPage = requests.get('https://www.icc-cricket.com/rankings/mens/player-rankings/odi/bowling')

webPage

bSoup = BeautifulSoup(webPage.content)

bSoup


# In[499]:


Men_Bowling_Rating = [] 

for i in bSoup.find_all('tr', class_='rankings-block__banner'):
    Men_Bowling_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n\n\n\n\n\n\n\n","#").replace("\n\n\n\n\n\n\n", "#").replace("\n\n\n\n\n","#").replace("n\n\n\n\n\n","").replace("\n\n\n", "#").replace("\n", "").strip().split("#"))
    
Men_Bowling_Rating


# In[500]:


for i in bSoup.find_all('tr', class_='table-body', limit = 9):
    Men_Bowling_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n","#").replace("\n\n\n\n", "#").replace("\n\n","#").replace("\n", "").strip().split("#"))
    
Men_Bowling_Rating


# In[501]:


Top_Ten_Bowler_Men = pd.DataFrame(data=Men_Bowling_Rating, columns=['None','Serial','Player_Name','Team','Rating','Best_Rate','None'])

Top_Ten_Bowler_Men


# In[503]:


Top_Ten_Bowler_Men.drop('None',axis='columns')


# In[504]:


#6) Write a python program to scrape cricket rankings from icc-cricket.com. You have to scrape and make data frame-
#a) Top 10 ODI teams in women’s cricket along with the records for matches, points and rating.
#b) Top 10 women’s ODI Batting players along with the records of their team and rating.
#c) Top 10 women’s ODI all-rounder along with the records of their team and rating.

webPage = requests.get('https://www.icc-cricket.com/rankings/womens/team-rankings/odi')

webPage

bSoup = BeautifulSoup(webPage.content)

bSoup


# In[505]:


Women_Cricket_List = []

for i in bSoup.find_all('tr', class_='rankings-block__banner'):
    Women_Cricket_List.append(i.text.replace("\n\n\n", "#").replace("\n\n","#").replace("\n","#").split("#"))
    
Women_Cricket_List


# In[510]:


for i in bSoup.find_all('tr', class_='table-body', limit=9):
   
        Women_Cricket_List.append(i.text.replace("\n\n\n", "#").replace("\n\n","#").replace("\n","#").split("#"))       
        
Women_Cricket_List


# In[511]:


Women_Top_ten_Cricket_Team = pd.DataFrame(data=Women_Cricket_List, columns =['None','Serial','Teams','Team_Abr','Matches','Point','Rating','None','None'])

Women_Top_ten_Cricket_Team


# In[512]:


Women_Top_ten_Cricket_Team.drop(['None','Team_Abr'],axis='columns', inplace=True)

Women_Top_ten_Cricket_Team


# In[513]:


#b) Top 10 women’s ODI Batting players along with the records of their team and rating.

webPage = requests.get('https://www.icc-cricket.com/rankings/womens/player-rankings/odi/batting')

webPage

bSoup = BeautifulSoup(webPage.content)

bSoup


# In[514]:


Women_Batting_Rating = [] 

for i in bSoup.find_all('tr', class_='rankings-block__banner'):
    Women_Batting_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n\n\n\n\n\n\n\n","#").replace("\n\n\n\n\n\n\n", "#").replace("\n\n\n\n\n","#").replace("n\n\n\n\n\n","").replace("\n\n\n", "#").replace("\n", "").strip().split("#"))
    
Women_Batting_Rating


# In[515]:


for i in bSoup.find_all('tr', class_='table-body', limit = 9):
    Women_Batting_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n","#").replace("\n\n\n\n", "#").replace("\n\n","#").replace("\n", "").strip().split("#"))
    
Women_Batting_Rating


# In[516]:


Top_Ten_Batwomen = pd.DataFrame(data=Men_Batmen_Rating, columns=['None','Serial','Player_Name','Team','Rating','Best_Rate','None'])

Top_Ten_Batwomen


# In[517]:


Top_Ten_Batwomen.drop(['None','Best_Rate'],axis='columns',inplace=True)


# In[518]:


Top_Ten_Batwomen


# In[519]:


# 6c) Top 10 women’s ODI all-rounder along with the records of their team and rating.

webPage = requests.get('https://www.icc-cricket.com/rankings/womens/player-rankings/odi/all-rounder')

webPage

bSoup = BeautifulSoup(webPage.content)

bSoup


# In[520]:


Women_AllRound_Rating = [] 

for i in bSoup.find_all('tr', class_='rankings-block__banner'):
    Women_AllRound_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n\n\n\n\n\n\n\n","#").replace("\n\n\n\n\n\n\n", "#").replace("\n\n\n\n\n","#").replace("n\n\n\n\n\n","").replace("\n\n\n", "#").replace("\n", "").strip().split("#"))
    
Women_AllRound_Rating


# In[521]:


for i in bSoup.find_all('tr', class_='table-body', limit = 9):
    Women_AllRound_Rating.append(i.text.replace("\n\n\n(0)\n\n\n\n\n","#").replace("\n\n\n\n", "#").replace("\n\n","#").replace("\n", "").strip().split("#"))
    
Women_AllRound_Rating


# In[522]:


Top_Ten_Women_AllRound_Rating = pd.DataFrame(data=Women_AllRound_Rating, columns=['None','Serial','Player_Name','Team','Rating','Best_Rate','None'])

Top_Ten_Women_AllRound_Rating


# In[523]:


Top_Ten_Women_AllRound_Rating.drop(['None','Best_Rate'],axis='columns',inplace=True)

Top_Ten_Women_AllRound_Rating


# In[533]:


# 7) Write a python program to scrape mentioned news details from https://www.cnbc.com/world/?region=world and make data frame-
# i) Headline 
# ii) Time
# iii) NewsLink


webPage = requests.get('https://www.cnbc.com/world/?region=world')

webPage

bSoup = BeautifulSoup(webPage.content, "html.parser")

bSoup


# In[534]:


News_Headlines_Time = [] # LatestNews-headlineWrapper    LatestNews-container

for i in bSoup.find_all('time', class_='LatestNews-timestamp'):
    News_Headlines_Time.append(i.text)#.replace("\n\n\n(0)\n\n\n\n\n\n\n\n\n\n\n\n","#").replace("\n\n\n\n\n\n\n", "#").replace("\n\n\n\n\n","#").replace("n\n\n\n\n\n","").replace("\n\n\n", "#").replace("\n", "").strip().split("#"))

News_Headlines_Time


# In[535]:


News_Headline = [] # LatestNews-headlineWrapper    LatestNews-container

for i in bSoup.find_all('a', class_='LatestNews-headline'):
    News_Headline.append(i.text)
    
News_Headline


# In[538]:


News_Link = []

#for i in bSoup.find_all('a', {'class':'LatestNews-headline'},href=True):
for i in bSoup.find_all('a', class_='LatestNews-headline'):
    News_Link.append(i.get('href'))
    
News_Link


# In[539]:


News_Headlines_Time_Link = pd.DataFrame({'Headlines': News_Headline, 'Time':News_Headlines_Time, 'Link':News_Link})

News_Headlines_Time_Link


# In[5]:


# 8) Write a python program to scrape the details of most downloaded articles from AI in last 90 days.https://www.journals.elsevier.com/artificial-intelligence/most-downloaded-articles Scrape below mentioned details and make data frame-
# i) Paper Title
# ii) Authors
# iii) Published Date
# iv) Paper URL

webPage = requests.get('https://www.journals.elsevier.com/artificial-intelligence/most-downloaded-articles')

webPage

bSoup = BeautifulSoup(webPage.content, "html.parser")

bSoup


# In[6]:


Articles_PaperTitle = []

for i in bSoup.find_all('h2', class_='sc-1qrq3sd-1 gRGSUS sc-1nmom32-0 sc-1nmom32-1 btcbYu goSKRg'):
    Articles_PaperTitle.append(i.text)
    
Articles_PaperTitle


# In[7]:


Articles_Authors = []

for i in bSoup.find_all('span', class_='sc-1w3fpd7-0 dnCnAO'):
    Articles_Authors.append(i.text)
    
Articles_Authors


# In[8]:


Articles_PublishedDate = []

for i in bSoup.find_all('span', class_='sc-1thf9ly-2 dvggWt'):
    Articles_PublishedDate.append(i.text)
    
Articles_PublishedDate


# In[9]:


Articles_Link = []

for i in bSoup.find_all('a', class_='sc-5smygv-0 fIXTHm'):
    Articles_Link.append(i.get('href'))
    
Articles_Link


# In[10]:


Articles_PTitle_Authors_Date_Link = pd.DataFrame({'Paper_Title': Articles_PaperTitle,'Authors':Articles_Authors,'Published_Date':Articles_PublishedDate,'Paper_URL':Articles_Link})

Articles_PTitle_Authors_Date_Link


# In[11]:


# 9) Write a python program to scrape mentioned details from dineout.co.in and make data frame-
# i) Restaurant name
# ii) Cuisine
# iii) Location
# iv) Ratings
# v) Image URL

#I can access the website, I'm blocked due to my location

webPage = requests.get('https://www.dineout.co.in/')

webPage

bSoup = BeautifulSoup(webPage.content, "html.parser")

bSoup


# In[ ]:




