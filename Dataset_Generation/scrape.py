from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import json
import jmespath
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import tiktoken
import re
import pandas as pd
import numpy as np
import os

df = pd.DataFrame(columns=['page_title', 'page_url', 'content'])

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class PGChunk:
    def __init__(self, title, url, content, content_length, content_tokens, embedding):
        self.page_title = title
        self.page_url = url
        self.content = content
        self.content_length = content_length
        self.content_tokens = content_tokens
        self.embedding = embedding

class PGPage:
    def __init__(self, title, url, content, length, tokens, chunks):
        self.title = title
        self.url = url
        self.content = content
        self.length = length
        self.tokens = tokens
        self.chunks = chunks

def get_title(driver):
    try :
        directory = driver.find_elements(By.CSS_SELECTOR, "script[type='application/ld+json']") 
        directory_string = []
        for dir in directory:
            directory_string.append(dir.get_attribute("innerHTML"))
        directory_string = '{}'.join(directory_string)
        directory_dict = json.loads(directory_string)
        path = jmespath.search('itemListElement[].name', directory_dict)
        return ' \\ '.join(path)
    except:
        try :
            return driver.find_elements(By.TAG_NAME, 'h1')[0].text
        except :
            return driver.current_url.split('https://docs.aws.amazon.com/')[1]
    

def get_content(driver):
    lines = []
    for title in driver.find_elements(By.TAG_NAME, 'h1'):
        curr = title.text
        pos = title.location['y']
        if curr:
            curr = curr.strip()
            curr = "@@@^^^" + curr + "^^^@@@"
            lines.append((curr, pos))

    for subtitle in driver.find_elements(By.TAG_NAME, 'h2'):
        curr = subtitle.text
        pos = subtitle.location['y']
        if curr:
            curr = curr.strip()
            curr = "@@@^^" + curr + "^^@@@"
            lines.append((curr, pos))
    
    for subsubtitle in driver.find_elements(By.TAG_NAME, 'h3'):
        curr = subsubtitle.text
        pos = subsubtitle.location['y']
        alpha = re.search('[a-zA-Z]', curr)
        if curr and alpha:
            curr = curr.strip()
            curr = "@@@^" + curr + "^@@@"
            lines.append((curr, pos))

    for para in driver.find_elements(By.TAG_NAME, 'p'):
        curr = para.text
        pos = para.location['y']
        if curr:
            curr = curr.strip()
            if curr and len(curr) > 0:
                if (curr[-1] != '.'):
                    curr = curr + "."
                lines.append((curr, pos))

    for code in driver.find_elements(By.TAG_NAME, 'pre'):
        curr = "".join(code.text.split())
        pos = code.location['y']
        alpha = re.search('[a-zA-Z]', curr)
        if curr and alpha:
            curr = curr.strip()
            curr = "@@@~~" + curr + "~~@@@"
            lines.append((curr, pos))

    lines.sort(key = lambda x: x[1])
    if len(lines) > 0:
        lines = list(zip(*lines))[0]
        return ' '.join(lines)
    else:
        return None


def chunk_page(content):
    CHUNK_SIZE = 200
    CHUNK_MAX = 250
    page_text_chunks = [];
    if num_tokens_from_string(content, "cl100k_base") > CHUNK_SIZE:
        split = '@@@'.join(content.split('. ')).split('@@@')
        chunkText = ""
        for sentence in split:
            sentence = sentence.strip()
            if len(sentence) == 0: 
                continue
            sentence_tokens = num_tokens_from_string(sentence, "cl100k_base");
            if sentence_tokens > CHUNK_SIZE:
                continue
            chunk_tokens = num_tokens_from_string(chunkText, "cl100k_base");
            if chunk_tokens + sentence_tokens > CHUNK_SIZE:
                page_text_chunks.append(chunkText.strip());
                chunkText = "";
            if re.search('[a-zA-Z]', sentence[-1]):
                chunkText += sentence + '. '
            else:
                chunkText += sentence + ' '
        page_text_chunks.append(chunkText.strip());
    else:
        page_text_chunks.append(content.strip())
    
    if len(page_text_chunks) > 2:
        last_elem = num_tokens_from_string(page_text_chunks[-1], "cl100k_base")
        second_to_last_elem = num_tokens_from_string(page_text_chunks[-2], "cl100k_base")
        if last_elem + second_to_last_elem < CHUNK_MAX:
            page_text_chunks[-2] += page_text_chunks[-1]
            page_text_chunks.pop()
    
    return page_text_chunks

def make_page(driver, url):
    global df
    driver.get(url)
    title = get_title(driver)
    content = get_content(driver)
    if content == None:
        return
    page_text_chunks = chunk_page(content)
    print(page_text_chunks.__len__())
    for chunk in page_text_chunks:
        df = pd.concat([df, pd.DataFrame({'page_title': [title], 'page_url': [url], 'content': [chunk]})])

        

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--incognito")
    options.add_argument("--disable-site-isolation-trials") 
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))


    print('Starting the Collection ... ')

    pages = open("./pages.txt", "r")
    for url in pages.readlines():
        make_page(driver, url.rstrip())
    pages.close()
        
    df.to_csv("./../data/dataset.csv", index=False)

if __name__ == "__main__":
    main()