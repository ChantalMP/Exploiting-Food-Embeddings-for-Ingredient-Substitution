import json
from typing import List

import requests


def scrap_food_recipe_reviews(url: str) -> List[str]:
    foodcom_base_url: str = 'https://api.food.com/external/v1/recipes/{}/feed/?pn={}&_=1587813097187&size=20&sort=-time'
    recipe_id: str = url.rsplit('-', 1)[-1]
    comments: List[str] = []
    try:
        for i in range(1, 50):  # Afterwards just stop
            current_page: str = foodcom_base_url.format(recipe_id, i)
            r = requests.get(current_page)
            as_json = json.loads(r.text)
            items = as_json['data']['items']

            if len(items) == 0:  # Reached limit of comments/reviews for that recipe, stop
                break

            for item in items:
                try:
                    comment: str = item['text'].replace('&amp;', '&')
                    comments.append(comment)

                except Exception as e:
                    pass

    except Exception as e:
        pass

    return comments
