import requests
from bs4 import BeautifulSoup


class ReviewScraperHTML:
    def __init__(self):
        pass

    def parse_html(self, url):
        r = requests.get(url)
        html = r.content.decode()

        parsed_html = BeautifulSoup(html, features="html.parser")
        return parsed_html

    def get_comments_cookeatshare(self, url):
        parsed_html = self.parse_html(url)
        review_texts = [elem.text.strip() for elem in
                        parsed_html.find_all('div', attrs={'class': 'text', 'itemprop': 'description'})]
        comments = [elem for elem in parsed_html.find_all('div', attrs={'class': 'comment message'})]
        comment_texts = []
        for comment in comments:
            comment_text = comment.find('div', attrs={'class': 'text'}).text.strip()
            comment_texts.append(comment_text)

        all_comments = review_texts + comment_texts
        return all_comments

    def get_comments_recipeland(self, url):
        parsed_html = self.parse_html(url)
        comment_texts = [elem.text.strip() for elem in parsed_html.body.find_all('div', attrs={'class': 'speech left'})]
        return comment_texts

    # here we only get 5 reviews even if there are more on the page
    def get_comments_foodandwine(self, url):
        parsed_html = self.parse_html(url)
        comment_texts = [elem.contents[3].text.replace("Review Body: ", "") for elem in parsed_html.body.find_all('div', attrs={'id': "review_"})]
        return comment_texts

