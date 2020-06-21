from enum import Enum
import re
import string
import torch

class WordEnum(Enum):
    PAD_TOKEN = 0
    SOS_TOKEN = 1
    EOS_TOKEN = 2

# Mapping index to word and vice versa
# SOS: start of sentence, EOS: end of sentence, PAD: Padding token\
# the model need to know what's the start and the end of each sentence)
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.index2word = {WordEnum.PAD_TOKEN: "PAD", WordEnum.SOS_TOKEN: "SOS", WordEnum.EOS_TOKEN: "EOS"}
        #self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.wordcount = {}
        self.number_words = 3

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.number_words
            self.index2word[self.number_words] = word
            self.wordcount[word] = 1
            self.number_words += 1
        else:
            self.wordcount[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # this function will create a new dictionnary that contains only words that appear more than a fixed threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        else:
            keep_words = []
            for k, v in self.wordcount.items():
                if v >= min_count:
                    keep_words.append(k)

            self.word2index = {}
            self.index2word = {WordEnum.PAD_TOKEN: "PAD", WordEnum.SOS_TOKEN: "SOS", WordEnum.EOS_TOKEN: "EOS"}
            self.wordcount = {}
            self.number_words = 3

            for word in keep_words:
                self.addWord(word)


def normalize_string(s):
    #the choice of this operations depends on the language
    #uncomment the functions that you need it in your context

    # remove numbers
    output = re.sub(r'\d+','', s)
    # remove punctuation
    output_str = output.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # remove urls
    output = re.sub(r"http\S+", "", output)
    #remove emails
    output = re.sub(r"\w+@\w+\.[a-z]{3}", "", output)
    #let only one space between words and transform all string to lowercase form
    output = " ".join(output.lower().split())
    return output

def indexesFromSentence(voc, sentence):

    indexSentence = []
    for word in normalize_string(sentence).split(' '):
        if word in voc.word2index:
            indexSentence.append(voc.word2index[word])
        else:
            continue
    return torch.LongTensor(indexSentence)


def maxLengthBatchSentences(batch_sentence):

    return max([len(elem) for elem in batch_sentence])


def LengthBatchSentences(batch_sentence):

    return torch.LongTensor(sorted([len(elem) for elem in batch_sentence], reverse = True))


def padTruncateSentenceToMax(maxLengthSentence, batchSentence):

    padded_sentence = torch.zeros(len(batchSentence), maxLengthSentence, dtype=torch.int64)
    for i in range(0, len(batchSentence)):
        if(maxLengthSentence >= len(batchSentence[i])):
            for j in range(0,len(batchSentence[i])):
                padded_sentence[i][j] = batchSentence[i][j]
        else:
            for j in range(0,maxLengthSentence):
                padded_sentence[i][j] = batchSentence[i][j]

    return padded_sentence

