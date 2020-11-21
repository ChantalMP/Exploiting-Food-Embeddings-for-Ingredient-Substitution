## Summary
Collect comments to the recipes in the Recipe1M+ dataset.
We chose to use urls from the following recipe pages: cookeatshare, food.com, foodandwine, recipeland, tastykitchen

## How it Works 
There are different parsers depending on the website. Some of them are just downloading the HTML page, whereas others are also making js queries.

## How to Run
Run     
    
    python -m comment_scraping.collect_comments
    
It will take a long time. Results will be saved in comment_scraiping/data in different jsons.

WARNING: These scripts are not maintained, so if changes are made to the websites, the respective parsers might not work anymore.
