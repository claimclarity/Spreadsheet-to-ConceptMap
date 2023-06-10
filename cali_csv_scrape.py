import pandas as pd
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def get_treatment_data(csv,pdf,length):
    """
    Uses NLTK backend NLP processing to gather sentences in which reccomendatiosn are being made.
    It then takes these and stores them in a column. It stores the treatment and rec together so you must manually 
    seperate them. Overall, not the most efficient thing but it sped up processing time by 10fold considering we used 
    to have to manually input all of this 
    :param csv: give output csv file name in form "example.csv"
    :param pdf: give input pdf file name in form "example.pdf"
    :param length: length of document; how many pages on it contain treatments
    :return: None
    """
    df = pd.read_csv(csv)
    doc = open(pdf,'rb')
    reader = PyPDF2.PdfReader(doc)
    page=0
    treat_sentences = []
    rec_pages = []
    while page<length:
        data = reader.pages[page].extract_text()
        page+=1
        sentences = sent_tokenize(data)
        for i in range(0,len(sentences)):
            if "Recommendation:" in sentences[i]:
                treat_sentences.append(str(sentences[i]))
                rec_pages.append(page)
    df['Treatment'] = pd.Series(treat_sentences)
    df['Page'] = pd.Series(rec_pages)
    df.to_csv(csv,index=False)
