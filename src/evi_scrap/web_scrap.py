# import pandas as pd
# from bs4 import BeautifulSoup
# from selenium import webdriver
# # driver = webdriver.Chrome(executable_path='/nix/path/to/webdriver/executable')
# options = webdriver.ChromeOptions()
# options.add_argument("--start-maximized")
# options.add_argument('--log-level=3')
# # Provide the path of chromedriver present on your system.
# driver = webdriver.Chrome(executable_path="chromedriver", chrome_options=options)
# driver.set_window_size(1920,1080)
# driver.get('https://www.psychologytoday.com/us/blog/close-encounters/201410/6-myths-about-men-women-and-relationships')
# results = []
# content = driver.page_source
# soup = BeautifulSoup(content)
# for element in soup.findAll(attrs={'class': 'title'}):
#     name = element.find('a')
#     results.append(name.text)


# for x in results:
#     print(x)

# df = pd.DataFrame({'text': results})
# df.to_csv('fact_pool.csv', index=False, encoding='utf-8')

# '//*[@id="block-pt-content"]/article/div[1]/div[2]/h1'

# from msilib.schema import tables
import requests
from bs4 import BeautifulSoup

# Make a request
page = requests.get(
    "https://www.psychologytoday.com/us/blog/close-encounters/201410/6-myths-about-men-women-and-relationships")
soup = BeautifulSoup(page.content, 'html5lib')#html.parser

# # Extract title of page
# page_title = soup.title.text

# # Extract body of page
# page_body = soup.body

# # Extract head of page
# page_head = soup.head

# # print the result
# print(page_body, page_head)

table = soup.findAll('h1', attrs={'class':'content-heading'})

for row in table:
    print(row.text)
