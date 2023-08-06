import requests

from bs4 import BeautifulSoup as bs
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

def main():
    url = 'https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom'

    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    text = soup.get_text()
    text = text.replace('\n', '')

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    return True

if __name__ == '__main__':
    main()
