'''
Collect comments to the recipes in the Recipe1m dataset
We chose to use urls from the following recipe pages: cookeatshare, food.com, foodandwine, recipeland, tastykitchen
Outputs: recipe1m_with_reviews.json
'''

import json
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from comment_scraping.scrapers import scrape_foodcom
from comment_scraping.scrapers import scrape_tastykitchen
from comment_scraping.scrapers.scrape_html_pages import ReviewScraperHTML


def get_reviews(url):
    reviews = []
    try:
        website_name = urlparse(url).netloc
        if website_name == 'cookeatshare.com':
            reviews = review_scraper_html.get_comments_cookeatshare(url)
        elif website_name == 'www.food.com':
            reviews = scrape_foodcom.scrap_food_recipe_reviews(url)
        elif website_name == 'www.foodandwine.com':
            reviews = review_scraper_html.get_comments_foodandwine(url)
        elif website_name == 'recipeland.com':
            reviews = review_scraper_html.get_comments_recipeland(url)
        elif website_name == 'tastykitchen.com':
            reviews = scrape_tastykitchen.scrap_food_recipe_reviews(url)
        else:
            raise Exception('got recipe with unaccepted url')
    except Exception as e:
        print(f'Url: {url} Exception {e}')
        return None
    return reviews


def filter_recipe1m_data_after_url(data):
    accepted_websites = ['cookeatshare.com', 'www.food.com', 'www.foodandwine.com', 'recipeland.com', 'tastykitchen.com']
    return [elem for elem in data if urlparse(elem['url']).netloc in accepted_websites]


def enhance_recipe(recipe):
    sep = " "
    ingredients = [d['text'] for d in recipe['ingredients']]
    instructions = sep.join([d['text'] for d in recipe['instructions']])
    url = recipe['url']
    id = recipe['id']
    reviews = get_reviews(url)
    if reviews is None:
        return None
    return {'id': id, 'url': url, 'ingredients': ingredients, 'instructions': instructions, 'reviews': reviews,
            'review_count': len(reviews)}


def enhance_recipe_1m_data():
    output_file = Path('comment_scraping/data/recipe1m_with_reviews.json')
    if output_file.exists():
        with output_file.open('r') as f:
            data_with_reviews = json.load(f)
    else:
        data_with_reviews = []

    with open('../data/recipe1m.json') as file:
        data = json.load(file)
        data = filter_recipe1m_data_after_url(data)

    processed_ids = {elem['id'] for elem in data_with_reviews}
    data = [elem for elem in data if elem['id'] not in processed_ids]
    print(len(data))
    print("loaded recipe1m data")

    chunksize = 2000
    chunked_data = [data[x:x + chunksize] for x in range(0, len(data), chunksize)]
    for chunk in tqdm(chunked_data, desc='Chunks'):
        chunk_reviews = thread_map(enhance_recipe, chunk, max_workers=1000, total=len(chunk), desc='Processing Chunk')
        chunk_reviews = [review for review in chunk_reviews if review]
        data_with_reviews.extend(chunk_reviews)
        with output_file.open('w') as f:
            json.dump(data_with_reviews, f)

    print("Finished scraping comments.")


if __name__ == '__main__':
    review_scraper_html = ReviewScraperHTML()
    enhance_recipe_1m_data()
