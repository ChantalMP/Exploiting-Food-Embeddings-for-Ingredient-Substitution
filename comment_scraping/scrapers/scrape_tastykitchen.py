from typing import List

import requests
from bs4 import BeautifulSoup


def html_parsing(parsed_html):
    comments = parsed_html.select("div[class^=comment\ byuser]")
    reviews = parsed_html.select("div[class^=review\ byuser]")
    all_comments = []
    for comment in comments:
        all_comments.append(comment.text.split('\n\n', 2)[-1].strip())
    for comment in reviews:
        all_comments.append(comment.text.split('\n\n\n', 1)[-1].strip())

    return all_comments


def api_parsing(comments_div, url):
    tastykitchen_base_url: str = 'https://tastykitchen.com/recipes/wp-admin/admin-ajax.php?action=more_comments&post_id={}&comment_type={}&page={}'
    data_post_id = comments_div.attrs['data-post-id']
    comments: List[str] = []
    try:
        for type in ['comment', 'review']:
            for i in range(1, 50):  # Afterwards just stop
                current_page: str = tastykitchen_base_url.format(data_post_id, type, i)
                as_json = requests.get(current_page).json()

                if as_json['success'] == False:
                    break

                html_string = as_json['html']
                parsed_html = BeautifulSoup(html_string, features="html.parser")
                comment_divs = parsed_html.findAll('div', attrs={'class': 'comment-text'})
                comments.extend([comment.text.split('\n\n', 2)[-1].strip() for comment in comment_divs])

    except Exception as e:
        print(current_page, '\n', url, e)

    return comments


def scrap_food_recipe_reviews(url: str) -> List[str]:
    r = requests.get(url)
    html = r.content.decode()
    parsed_html = BeautifulSoup(html, features="html.parser")
    comments_div = parsed_html.find('div', attrs={'class': 'load-more js-load-more-comments'})

    if comments_div is None:
        comments = html_parsing(parsed_html)

    else:
        comments = api_parsing(comments_div, url)

    return comments
