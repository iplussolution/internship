#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install selenium')


# In[449]:


import selenium as sl
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})


# In[6]:


#Q1: Write a python program to scrape data for “Data Analyst” Job position in “Bangalore” location. 
#You have to scrape the job-title, job-location, company_name, experience_required. You have to scrape first 10 jobs data.

#get the webpage https://www.naukri.com/
driver.get('https://www.naukri.com/')

#create variable for designation field, get and assign the field address
designation = driver.find_element(By.CLASS_NAME, "suggestor-input ")
#set value(Data Analyst) into designation field
designation.send_keys('Data Analyst')


# In[7]:


#create variable for location field, get and assign the field address
location = driver.find_element(By.XPATH, "/html/body/div/div[7]/div/div/div[5]/div/div/div/div[1]/div/input")
#set value(Bangalore) into designation field
location.send_keys('Bangalore')


# In[8]:


#create variable for search button, get and assign the address
search = driver.find_element(By.CLASS_NAME,"qsbSubmit")

#click search button
search.click()


# In[9]:


#creating variables to hold list of job titles, location, company name and required experience
job_title = []
job_location = []
company_name = []
experience_required = []


# In[10]:


# Scraping top 10 job title from the given page
title_tags = driver.find_elements(By.XPATH,"//a[@class='title ellipsis']")
for i in title_tags[0:10]:
    title = i.text
    job_title.append(title)  


# In[11]:


# Scraping top 10 location from the given page
location_tags = driver.find_elements(By.XPATH,"//span[@class='ellipsis fleft locWdth']")
for i in location_tags[0:10]:
    title = i.text
    job_location.append(title)


# In[12]:


# Scraping top 10 company from the given page
company_tags = driver.find_elements(By.XPATH,"//a[@class='subTitle ellipsis fleft']")
for i in company_tags[0:10]:
    company = i.text
    company_name.append(company)


# In[13]:


# Scraping top 10 experiences from the given page
experience_tags = driver.find_elements(By.XPATH,"//span[@class='ellipsis fleft expwdth']")
for i in experience_tags[0:10]:
    exp = i.text
    experience_required.append(exp)


# In[14]:


#loading the list in dataframe
df = pd.DataFrame({'Title':job_title,'Location':job_location,'Company':company_name,'Experience':experience_required})
df


# In[19]:


#Q2: Write a python program to scrape data for “Data Scientist” Job position in “Bangalore” location. 
#    You have to scrape the job-title, job-location, company_name. You have to scrape first 10 jobs data.

#get the webpage https://www.naukri.com/
import selenium as sl
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})


driver.get('https://www.naukri.com/')

#create variable for designation field, get and assign the field address
designation = driver.find_element(By.CLASS_NAME, "suggestor-input ")
#set value(Data Scientist) into designation field
designation.send_keys('Data Scientist')


# In[20]:


#create variable for location field, get and assign the field address
location = driver.find_element(By.XPATH, "/html/body/div/div[7]/div/div/div[5]/div/div/div/div[1]/div/input")
#set value(Bangalore) into designation field
location.send_keys('Bangalore')


# In[21]:


#create variable for search button, get and assign the address
search = driver.find_element(By.CLASS_NAME,"qsbSubmit")

#click search button
search.click()


# In[22]:


#creating variables to hold list of job titles, location, company name and required experience
job_title = []
job_location = []
company_name = []


# In[23]:


# Scraping top 10 job title from the given page
title_tags = driver.find_elements(By.XPATH,"//a[@class='title ellipsis']")
for i in title_tags[0:10]:
    title = i.text
    job_title.append(title) 


# In[24]:


# Scraping top 10 location from the given page
location_tags = driver.find_elements(By.XPATH,"//span[@class='ellipsis fleft locWdth']")
for i in location_tags[0:10]:
    title = i.text
    job_location.append(title)


# In[25]:


# Scraping top 10 company from the given page
company_tags = driver.find_elements(By.XPATH,"//a[@class='subTitle ellipsis fleft']")
for i in company_tags[0:10]:
    company = i.text
    company_name.append(company)


# In[26]:


#loading the list in dataframe
df = pd.DataFrame({'Title':job_title,'Location':job_location,'Company':company_name})
df


# In[43]:


#Q3: In this question you have to scrape data using the filters available on the webpage as shown below:
#You have to use the location and salary filter.
#You have to scrape data for “Data Scientist” designation for first 10 job results.
#You have to scrape the job-title, job-location, company name, experience required.
#The location filter to be used is “Delhi/NCR”. The salary filter to be used is “3-6” lakhs

import selenium as sl
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.naukri.com/
driver.get('https://www.naukri.com/')

#create variable for designation field, get and assign the field address
designation = driver.find_element(By.CLASS_NAME, "suggestor-input ")
#set value(Data Scientist) into designation field
designation.send_keys('Data Scientist')


# In[44]:


#create variable for search button, get and assign the address
search = driver.find_element(By.CLASS_NAME,"qsbSubmit")

#click search button
search.click()


# In[45]:


#Code to select location
location_checkbox = driver.find_element(By.XPATH, "/html/body/div[1]/div[4]/div/div/section[1]/div[2]/div[5]/div[2]/div[2]/label/i")
location_checkbox.click()


# In[47]:


#Code to select salary 
salary_checkbox = driver.find_element(By.XPATH, "/html/body/div[1]/div[4]/div/div/section[1]/div[2]/div[6]/div[2]/div[2]/label/i")
salary_checkbox.click()


# In[48]:


#creating variables to hold list of job titles, location, company name and required experience
job_title = []
job_location = []
company_name = []
experience_required = []


# In[49]:


# Scraping top 10 job title from the given page
title_tags = driver.find_elements(By.XPATH,"//a[@class='title ellipsis']")
for i in title_tags[0:10]:
    title = i.text
    job_title.append(title) 


# In[50]:


# Scraping top 10 location from the given page
location_tags = driver.find_elements(By.XPATH,"//span[@class='ellipsis fleft locWdth']")
for i in location_tags[0:10]:
    title = i.text
    job_location.append(title)


# In[51]:


# Scraping top 10 company from the given page
company_tags = driver.find_elements(By.XPATH,"//a[@class='subTitle ellipsis fleft']")
for i in company_tags[0:10]:
    company = i.text
    company_name.append(company)


# In[52]:


# Scraping top 10 experiences from the given page
experience_tags = driver.find_elements(By.XPATH,"//span[@class='ellipsis fleft expwdth']")
for i in experience_tags[0:10]:
    exp = i.text
    experience_required.append(exp)


# In[53]:


#loading the list in dataframe
df = pd.DataFrame({'Title':job_title,'Location':job_location,'Company':company_name,'Experience':experience_required})
df


# In[450]:


#Q4: Scrape data of first 100 sunglasses listings on flipkart.com. You have to scrape four attributes: 
#1. Brand
#2. Product Description
#3. Price
#The attributes which you have to scrape is ticked marked in the below image.

import selenium as sl
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.flipkart.com/
driver.get('https://www.flipkart.com/')


# In[118]:


#create variable for designation field, get and assign the field address
products_brands_more = driver.find_element(By.CLASS_NAME, "_3704LK")
#set value(Data Scientist) into designation field
products_brands_more.clear()
products_brands_more.send_keys('sunglasses')


# In[120]:


#create variable for search button, get and assign the address
search = driver.find_element(By.CLASS_NAME,"L0Z3Pu")

#click search button
search.click()


# In[137]:


#Defining variables that will hold list of product brands, description,prices and discount
product_brands = []
product_descriptions =[]
product_prices = []
product_discounts = []


# In[138]:


#Code to scrape top 100 sunglasses brand,description,price and discount
x=0
y=0
z=0
u=0
while True:
    
    #Scraping top 100 brands
    brands=driver.find_elements(By.XPATH,"//div[@class='_2WkVRV']")
    for i in brands:
        product_brands.append(i.text)
        x+=1
        if x==100:
            break
            
    #Scraping top 100 product description
    desc =driver.find_elements(By.XPATH,"//a[@class='IRpwTa']")
    for i in desc:
        product_descriptions.append(i.text)
        y+=1
        if y==100:
            break   
            
    #Scraping top 100 prices
    price =driver.find_elements(By.XPATH,"//div[@class='_30jeq3']")
    for i in price:
        product_prices.append(i.text)
        z+=1
        if z==100:
            break
            
    #Scraping top 100 discounts
    discount =driver.find_elements(By.XPATH,"//div[@class='_3Ay6Sb']")
    for i in discount:
        product_discounts.append(i.text)
        u+=1
        if u==100:
            break
    if u == 100:
        break
    next_button = driver.find_element(By.XPATH,"//a[@class='_1LKTO3']")
    next_button.click()
    time.sleep(3)


# In[139]:


print(len(product_brands),len(product_descriptions),len(product_prices),len(product_discounts))


# In[141]:


#loading the list in dataframe
df = pd.DataFrame({'Brand':product_brands,'Description':product_descriptions,'Price':product_prices,'Discount':product_discounts})
df


# In[173]:


#Q5: Scrape 100 reviews data from flipkart.com for iphone11 phone. You have to go the link: https://www.flipkart.com/apple-iphone-11-black-64-gb/product- reviews/itm4e5041ba101fd?pid=MOBFWQ6BXGJCEYNY&lid=LSTMOBFWQ6BXGJCEYNYZXSHRJ&market place=FLIPKART
#As shown in the above page you have to scrape the tick marked attributes. These are:
#1. Rating
#2. Review summary
#3. Full review
#4. You have to scrape this data for first 100reviews.

import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})


# get the site
driver.get('https://www.flipkart.com/apple-iphone-11-black-64-gb/product-reviews/itm4e5041ba101fd?pid=MOBFWQ6BXGJCEYNY&lid=LSTMOBFWQ6BXGJCEYNYZXSHRJ&sortOrder=MOST_HELPFUL&certifiedBuyer=false&aid=overall')


# In[177]:


#variables to hold list of reviews rating, review summary, full reviews
review_ratings = []
review_summarys =[]
full_reviews = []


# In[178]:


#Code to scrape top 100 reviews rating,review summary,full review
x=0
y=0
z=0
while True:
    
    #Scraping top 100 ratings
    ratings=driver.find_elements(By.XPATH,"//div[@class='_3LWZlK _1BLPMq']")
    for i in ratings:
        review_ratings.append(i.text)
        x+=1
        if x==100:
            break
            
    #Scraping top 100 reviews
    summarys =driver.find_elements(By.XPATH,"//p[@class='_2-N8zT']")
    for i in summarys:
        review_summarys.append(i.text)
        y+=1
        if y==100:
            break  
            
    #Scraping top 100 full reviews        
    reviews =driver.find_elements(By.XPATH,"//div[@class='t-ZTKy']")
    for i in reviews:
        full_reviews.append(i.text)
        z+=1
        if z==100:
            break
    
    if x == 100:
        break
    next_button = driver.find_element(By.XPATH,"//a[@class='_1LKTO3']")
    next_button.click()
    time.sleep(3)


# In[179]:


print(len(review_ratings),len(review_summarys),len(full_reviews))


# In[180]:


#loading the list in dataframe
df = pd.DataFrame({'Review ratings':review_ratings,'Summary':review_summarys,'Full review':full_reviews})
df


# In[181]:


#Q6: Scrape data for first 100 sneakers you find when you visit flipkart.com and search for “sneakers” in the search field.
#You have to scrape 3 attributes of each sneaker:
#1. Brand
#2. Product Description
#3. Price

import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.flipkart.com/
driver.get('https://www.flipkart.com/')


# In[185]:


#create variable for search field, get and assign the field address
products_brands_more = driver.find_element(By.CLASS_NAME, "_3704LK")
#set value(sneakerst) into designation field
products_brands_more.clear()
products_brands_more.send_keys('sneakers')


# In[186]:


#create variable for search button, get and assign the address
search = driver.find_element(By.CLASS_NAME,"L0Z3Pu")

#click search button
search.click()


# In[187]:


#define variables for list of products brands, description and prices
product_brands = []
product_descriptions =[]
product_prices = []


# In[188]:


#Code to scrape top 100 sneakers brand,description,price and discount
x=0
y=0
z=0

while True:
    
    #Scraping top 100 brands
    brands=driver.find_elements(By.XPATH,"//div[@class='_2WkVRV']")
    for i in brands:
        product_brands.append(i.text)
        x+=1
        if x==100:
            break
            
    #Scraping top 100 product description
    desc =driver.find_elements(By.XPATH,"//a[@class='IRpwTa']")
    for i in desc:
        product_descriptions.append(i.text)
        y+=1
        if y==100:
            break   
            
    #Scraping top 100 prices
    price =driver.find_elements(By.XPATH,"//div[@class='_30jeq3']")
    for i in price:
        product_prices.append(i.text)
        z+=1
        if z==100:
            break
            
    if x == 100:
        break
    next_button = driver.find_element(By.XPATH,"//a[@class='_1LKTO3']")
    next_button.click()
    time.sleep(3)


# In[189]:


print(len(product_brands),len(product_descriptions),len(product_prices))


# In[190]:


#loading the list into dataframe
df = pd.DataFrame({'Brand':product_brands,'Description':product_descriptions,'Price':product_prices})
df


# In[451]:


#Q7: Go to webpage https://www.amazon.in/ Enter “Laptop” in the search field and then click the search icon. 
#Then set CPU Type filter to “Intel Core i7” as shown in the below image:


import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.amazon.in/
driver.get('https://www.amazon.in/')


# In[215]:


#create variable for search field, get and assign the field address
products_brands_more = driver.find_element(By.ID, "twotabsearchtextbox")
#set value(laptop) into the field
products_brands_more.clear()
products_brands_more.send_keys('laptop')


# In[216]:


#create variable for search button, get and assign the address
search = driver.find_element(By.ID,"nav-search-submit-button")

#click search button
search.click()


# In[217]:


#create variable for checkbox, get and assign the address
cpu = driver.find_element(By.XPATH,"/html/body/div[1]/div[2]/div[1]/div[2]/div/div[3]/span/div[1]/div/div/div[6]/ul[7]/span[9]/li/span/a/div/label/i")

#click the checkbox
cpu.click()


# In[220]:


#define variables for laptop titles, ratings and prices
labtop_title = []
laptop_ratings =[]
laptop_prices = []


# In[221]:


#Code to scrape top 10 laptops titles, ratings and prices
x=0
y=0
z=0

while True:
    
    #Scraping top 10 titles
    titles=driver.find_elements(By.XPATH,"//span[@class='a-size-medium a-color-base a-text-normal']")
    for i in titles:
        labtop_title.append(i.text)
        x+=1
        if x==10:
            break
            
    #Scraping top 10 ratings
    ratings =driver.find_elements(By.XPATH,"//span[@class='a-size-base s-underline-text']")
    for i in ratings:
        laptop_ratings.append(i.text)
        y+=1
        if y==10:
            break   
            
    #Scraping top 10 prices
    prices =driver.find_elements(By.XPATH,"//span[@class='a-price-whole']")
    for i in prices:
        laptop_prices.append(i.text)
        z+=1
        if z==10:
            break
            
    if x == 10:
        break
    next_button = driver.find_element(By.XPATH,"//a[@class='s-pagination-item s-pagination-next s-pagination-button s-pagination-separator']")
    next_button.click()
    time.sleep(3)


# In[223]:


#Checking the lenght of each list
print(len(labtop_title),len(laptop_ratings),len(laptop_prices))


# In[224]:


#loading the list into dataframe
df = pd.DataFrame({'Title':labtop_title,'Rating':laptop_ratings,'Price':laptop_prices})
df


# In[366]:


#Write a python program to scrape data for Top 1000 Quotes of All Time. The above task will be done in following steps:
#1. First get the webpage https://www.azquotes.com/ 
#2. Click on Top Quotes
#3. Than scrap a) Quote b) Author c) Type Of Quotes

import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.azquotes.com
driver.get('https://www.azquotes.com')


# In[377]:


#create variable for search button, get and assign the address
search = driver.find_element(By.XPATH,"/html/body/div[1]/div[2]/div[1]/div/div[3]/ul/li[5]/a")

#click search button
search.click()


# In[378]:


quotes = []
authors = []
quote_types = []


# In[379]:


#Code to scrape top 10 laptops titles, ratings and prices
x=0
y=0
z=0

while True:
    
    #Scraping top 1000 titles
    quo=driver.find_elements(By.XPATH,"//a[@class='title']")
    for i in quo:
        quotes.append(i.text)
        x+=1
        if x==1000:
            break
            
    #Scraping top 1000 ratings
    auth =driver.find_elements(By.XPATH,"//div[@class='author']")
    for i in auth:
        authors.append(i.text)
        y+=1
        if y==1000:
            break   
            
    #Scraping top 10 prices
    qType =driver.find_elements(By.XPATH,"//div[@class='tags']")
    for i in qType:
        quote_types.append(i.text)
        z+=1
        if z==1000:
            break
            
    if x == 1000:
        break
    next_button = driver.find_element(By.XPATH,"/html/body/div[1]/div[3]/div/div/div/div[1]/div/div[3]/li[12]/a")
    next_button.click()
    time.sleep(3)


# In[380]:


#Checking the lenght of each list
print(len(quotes),len(authors),len(quote_types))


# In[381]:


#loading the list into dataframe
df = pd.DataFrame({'Quote':quotes,'Author':authors,'Type':quote_types})
df


# In[274]:


#Q9: Write a python program to display list of respected former Prime Ministers of India(i.e. Name, Born-Dead, Term of office, Remarks) from https://www.jagranjosh.com/.
#This task will be done in following steps:
#1. First get the webpage https://www.jagranjosh.com/
#2. Then You have to click on the GK option
#3. Then click on the List of all Prime Ministers of India
#4. Then scrap the mentioned data and make theDataFrame.

import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.jagranjosh.com/
driver.get('https://www.jagranjosh.com/')


# In[277]:


#create variable for search button, get and assign the address
search = driver.find_element(By.XPATH,"/html/body/div/div[1]/div/div[1]/div/div[5]/div/div[1]/header/div[3]/ul/li[3]/a")

#click on GK option
search.click()


# In[278]:


#create variable for list of prime minister's link, get and assign the address
search = driver.find_element(By.XPATH,"/html/body/div[1]/div/div/div[2]/div/div[10]/div/div/ul/li[2]/a")

#click on link
search.click()


# In[279]:


primeM = []


# In[338]:


#Scraping Prime minister's name, Born-Death,Term of office.

#create variable and fetch table rows into it
table_trs = driver.find_elements(By.XPATH, '//div[@class="table-box"]/table/tbody/tr')

#create variable and fetch row columns into it to be sure of cols fetched
l=driver.find_elements(By.XPATH, "//*[@class= 'table-box']/table/tbody/tr[2]/td")

#create variable and get the lenght of the table
row = len(table_trs)

#print ot colums count
print(len(l))

#variable to hold list of prime ministers's name,Born-Death,Term of office
value_list = []

#loop through the table to fetch records for name,Born-Death,Term of office and save in list
for row in table_trs[1:row-2]:
    value_list.append({
        'Prime minister Name':row.find_elements(By.TAG_NAME, "td")[1].text,
        'Born-Death':row.find_elements(By.TAG_NAME, "td")[2].text,
        'Term of Office':row.find_elements(By.TAG_NAME, "td")[3].text,
        
    })

# Create Dataframe for the list
df = pd.DataFrame(value_list)

df


# In[382]:


#Q10: Write a python program to display list of 50 Most expensive cars in the world (i.e. Car name and Price) from https://www.motor1.com/
#This task will be done in following steps:
#1. First get the webpagehttps://www.motor1.com/
#2. Then You have to click on the List option from Dropdown menu on left side. 3. Then click on 50 most expensive cars in the world..
#4. Then scrap the mentioned data and make the dataframe.

import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options

#defining and setting driver options and agent to ensure compatibility with my Chrome version
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'/Users/opeyemiojeyinka/Documents/chromedriver_mac_arm64/chromedriver.exe')
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                     'Chrome/85.0.4183.102 Safari/537.36'})

#get the webpage https://www.motor1.com/
driver.get('https://www.motor1.com/')


# In[390]:


#create variable for harmburger, get and assign the address
search = driver.find_element(By.CLASS_NAME,"m1-hamburger-button")

#click on harmburger icon
search.click()


# In[405]:


#create variable for feature link, get and assign the address
search = driver.find_element(By.XPATH,"/html/body/div[9]/div[1]/div[3]/ul/li[5]/button")

#click on the link
search.click()


# In[408]:


#create variable for list link, get and assign the address 
search = driver.find_element(By.XPATH,"/html/body/div[9]/div[1]/div[3]/ul/li[6]/ul/li[5]/a")

#click on the link
search.click()


# In[411]:


#create variable for 20 most expensive cars link, get and assign the address 
search = driver.find_element(By.XPATH,"/html/body/div[8]/div[8]/div[1]/div[1]/div/div/div[9]/div/div[1]/h3/a")

#click on the link
search.click()


# In[444]:


cars = []
prices = []


# In[445]:


#Code to scrape quotes car name and price
    
#Scraping car
car=driver.find_elements(By.XPATH,"//h3[@class='subheader']")
for i in car:
    cars.append(i.text)
            
#Scraping price
price =driver.find_elements(By.TAG_NAME,'strong')
for a in price:
    prices.append(a.text) 


# In[446]:


#Checking the lenght of each list
print(len(cars),len(prices))


# In[448]:


#remove inappropriate value from car
cars.remove('Most Expensive Cars In The World')
#remove empty string from prices
prices.remove('')

#load records into dataframe
df = pd.DataFrame({'Cars':cars,'Prices':prices})
df


# In[ ]:




