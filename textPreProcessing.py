import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd

class DataProcessor():
   
    def __init__(self):

        nltk.download('wordnet')
        nltk.download('vader_lexicon')

        self.lemmatizer = WordNetLemmatizer()

        with open("./slang.txt",'r') as f:
            lines = f.readlines()
        
        self.chatWords = dict()
        for i in range(len(lines)):
            splitLine = lines[i].split('=')
            self.chatWords[splitLine[0]] = splitLine[1].replace("\n", "")

        self.exclude = string.punctuation

    # Remove HTML tags
    def removeHtmlTags(self, text):
        pattern = re.compile('<.*?>')
        return re.sub(pattern, r'', text)

    # Translate emojies to be more friendly for neural networksx
    def handleEmojis(self, text):

        emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
            ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
            ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
            ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
            '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
            '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
            ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

        for emoji in emojis.keys():
            text = text.replace(emoji, "EMOJI" + emojis[emoji])

        return text

    def translateSlang(self, text):
        returnString = text

        for word in text.split():
            if word.upper() in self.chatWords:
                returnString = returnString.replace(word, self.chatWords[word.upper()])

        return returnString

    def removeUsername(self, text):
        at = re.compile(r'@[A-Za-z0-9_]+')
        return at.sub(r'',text)

    def removeURL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def removePunc(self, text):
        return text.translate(str.maketrans('', '', self.exclude))

    def removeLeadingSpace(self, text):
        return text.lstrip()

    def lemmatizeText(self, text):

        lemmatizedTokens = ([self.lemmatizer.lemmatize(w.lower()) for w in text.split()])

        return " ".join(lemmatizedTokens)

    def CleanTextData(self, dataframe):
    
        dataframe = dataframe.apply(lambda x: self.removeHtmlTags(x))
        dataframe = dataframe.apply(lambda x: self.removeUsername(x))
        dataframe = dataframe.apply(lambda x: self.removeURL(x))
        dataframe = dataframe.apply(lambda x: self.translateSlang(x))
        dataframe = dataframe.apply(lambda x: self.removePunc(x))
        dataframe = dataframe.apply(lambda x: self.removeLeadingSpace(x))
        dataframe = dataframe.apply(lambda x: self.lemmatizeText(x))

        return dataframe
    