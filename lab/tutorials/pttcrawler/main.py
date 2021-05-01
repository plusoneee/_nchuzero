import requests
from bs4 import BeautifulSoup
import csv
url = 'https://www.imdb.com/chart/top?ref_=nv_mv_250'


def to_csv(table):
  with open('output.csv', 'a+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table)

def get_top_movie(url):
  headers = {"Accept-Language": "en-US,en;"}
  res = requests.get(url, headers=headers)

  tables = list()

  if res.status_code == requests.codes.ok:  # 200
    soup = BeautifulSoup(res.text, 'lxml')

  # CSS select
  lister = soup.select('tbody.lister-list tr') # list
  for idx, td in enumerate(lister):
    title = td.select_one('td.titleColumn a').text
    href = td.select_one('td.titleColumn a').get('href')
    movie_url = 'https://www.imdb.com' + href
    year = td.select_one('td.titleColumn span').text
    rating_body = td.select_one('td.ratingColumn strong')
    rating_description = rating_body.get('title')
    rating_number = rating_body.text
    detail_of_movie_dict = get_movie_detail(movie_url)

    tables.append([
      title, year, rating_number, rating_description, movie_url, detail_of_movie_dict
    ])

    if idx + 1 % 10 == 0:
      to_csv(tables)
      tables = list()
      break
    # print(idx + 1, title, year, rating_number, '(', rating_description, ')', movie_url, detail_of_movie_dict)


def get_movie_detail(movie_url):
  headers = {"Accept-Language": "en-US,en;"}
  res = requests.get(movie_url, headers=headers)
  if res.status_code == requests.codes.ok:  # 200
    soup = BeautifulSoup(res.text, 'lxml')
  summary = soup.select_one('div.plot_summary')
  summary_text = summary.select_one('div.summary_text').text.strip()
  credits_items = summary.select('div.credit_summary_item')

  item_dict = dict()
  for item in credits_items:
    inline = item.select_one('h4.inline').text[:-1]
    members = item.select('a')
    item_dict[inline] = list()

    for member in members:
      item_dict[inline].append(member.text)

  item_dict['summary_text'] = summary_text
  return item_dict


get_top_movie(url)







