import re, time
import requests
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class Scraper:
    def __init__(self, tag="trump", limit=1000):
        self.url = f"https://mastodon.social/api/v1/timelines/tag/{tag}"
        self.limit = limit
        self.posts = []

    def run(self):
        max_id = None
        while len(self.posts) < self.limit:
            params = {'limit': 50}
            if max_id:
                params['max_id'] = max_id

            try:
                r = requests.get(
                    url=self.url,
                    params=params
                )
                if r.status_code == 200:
                    data = r.json()
                    logger.info("200 OK")
                    if not data:
                        break
                    for post in data:
                        if post.get('language') == "en" :
                            content = re.sub('<[^<]+?>', '', post['content'])
                            if len(content) > 30:
                                self.posts.append({
                                    "created_at": post['created_at'],
                                    "username": post['account']['username'],
                                    "text": content.replace("\n", " ").strip()
                                })
                    max_id = data[-1]['id']
                    time.sleep(0.3)
                else:
                    logger.error(f"Error while requesting: {r.status_code}")
                    break
            except Exception as e:
                logger.error(f"Request failed: {e}")
                break
        
        return pd.DataFrame(self.posts)

    def save_to_csv(self, df, file_name="data/raw/scraped-data.csv"):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        df.to_csv(file_name, index=False, encoding='utf-8')

if __name__ == "__main__":
    import os
    scraper = Scraper()
    df = scraper.run()
    scraper.save_to_csv(df)
