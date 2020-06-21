from vocabulary_indexation import *
from model_architecture import EncoderRNN
import torch.nn as nn


def main():

    # create a vocabulary
    voc = Voc('english')
    voc.addSentence(normalize_string("You should add your sentence, then it will be normalized using the function created in vocabulary indexation file"))
    print(voc.word2index)

    #sentences to feed to the model
    sentence1 = indexesFromSentence(voc, "You should add your sentence")
    sentence2 = indexesFromSentence(voc, " it will be normalized using the function")
    sentence3 = indexesFromSentence(voc, "vocabulary indexation file")

    #feed the model a batch of sentences
    L = []
    L.append(sentence1)
    L.append(sentence2)
    L.append(sentence3)
    max_length_sentence = maxLengthBatchSentences(L)
    length_sentences = LengthBatchSentences(L)

    #create an instance of a model
    encoder = EncoderRNN(input_size = 7807, hidden_size = 512, token_numbers = max_length_sentence ,embedding= nn.Embedding(len(voc.index2word), 7807, padding_idx=0))

    #add the padding token to sentences
    padded_sentences = padTruncateSentenceToMax(max_length_sentence, L)

    #forward pass
    outputs_level1_encoding, outputs_level2_encoding, outputs_level3_encoding = encoder.forward(padded_sentences, length_sentences)

    #print shape of encoding vectors
    print(outputs_level1_encoding.shape)
    print(outputs_level2_encoding.shape)
    print(outputs_level3_encoding.shape)


if __name__ == "__main__":
    main()

