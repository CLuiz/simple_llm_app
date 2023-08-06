import requests

from bs4 import BeautifulSoup as bs
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

def get_data(url = 'https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom'):
    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    text = soup.get_text()
    text = text.replace('\n', '')

    with open('output.txt', 'w', encoding='utf-8') as file:
        print('writing text')
        file.write(text)
    return True

def split_text(doc="output.txt"):

    with open('output.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 100,
            length_function = len
            )
    texts = text_splitter.create_documents([text])

    return texts

def main():
    get_data()
    texts = split_text()

    return texts

def set_embeddings(texts, embeddings = OpenAIEmbeddings()):

    db = Chroma.from_documents(texts, embeddings)

    return db

if __name__ == '__main__':
    texts = main()
    db = set_embeddings(texts=texts)


