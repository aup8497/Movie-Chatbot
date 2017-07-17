import random
import pickle
import nltk

''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines=open('./data/movie_lines.txt').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open('./data/movie_conversations.txt').read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs

'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''
def extract_conversations(convs,id2line,path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx)+'.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []
    # here questions are encoding data to be sent to the seqseq model
    for conv in convs:
        if len(conv) %2 != 0:   # if the conversation has odd number of terms , then remove the last one so that we can assign odd sentences as questions and even sentences as answer
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


'''
    We need 4 files
    1. train.enc : Encoder input for training
    2. train.dec : Decoder input for training
    3. test.enc  : Encoder input for testing
    4. test.dec  : Decoder input for testing
'''
def prepare_seq2seq_files(questions, answers, path='',TESTSET_SIZE = 30000):
    
    # open files
    train_enc = open(path + 'train.enc','w')
    train_dec = open(path + 'train.dec','w')
    test_enc  = open(path + 'test.enc', 'w')
    test_dec  = open(path + 'test.dec', 'w')

    # choose 30,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))],TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i%10000 == 0:
            print('\n>> written %d lines' %(i)) 

    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
    

def create_data_set_and_load_into_pickle():
    id2line = get_id2line()
    #  here id2line = {  'L487031': "Just relax. You're at school now. No one can get you here.", 'L487030': 'Not this time. I owe Cotton that much. Hell, even I thought that man was guilty.', 'L487037': "This is a mistake. I shouldn't be here." }
    print( '>> gathered id2line dictionary.\n')
    convs = get_conversations()
    # here convs = [ ['L665932', 'L665933'], ['L665936', 'L665937'], ['L665938', 'L665939']]
    print ('>> gathered conversations.\n')
    questions, answers = gather_dataset(convs,id2line) # here questions and answers are every two consecutive sentences in the big dataset of conversations
    # here questios and answers are in the form => 'Can we make this quick?.'

    # storing these sentences in a pickle

    data ={}
    data['questions'] = questions
    data['answers'] = answers

    print ('>> writing questions and answers data into data.pkl\n')
    with open('data.pkl','wb') as fp:
        pickle.dump(data,fp)

    # you can load the dict using 
    # with open("test.txt", "rb") as fp:   # Unpickling
    #   b = pickle.load(fp)


def getID(word, vocab, create=True):
    word = word.lower()
    wid = vocab["word2id"].get(word,vocab, -1)
    if wid == -1:
        if create:
            wid = len(vocab["word2id"])
            vocab["word2id"][word] = wid
        else:
            wid = vocab["word2id"].get("<unknown>")
    return wid


def generate_vocab():

    with open("data.pkl", "rb") as fp:   # Unpickling
        data = pickle.load(fp)


    print('Creating vocabulary...')
    vocab = {}
    vocab["word2id"] = {}
    vocab["id2word"] = {}

    getID('<go>',vocab)
    getID('<pad>',vocab)
    getID('<eos>',vocab)
    getID('<unknown>',vocab)

    for sentence in data['questions']:
        for word in nltk.word_tokenize(sentence.decode('ISO-8859-1')):
            getID(word,vocab)

    vocab["id2word"] = { v: k for k, v in vocab["word2id"].items() }
    print('Vocabulary created.Inverse vocabulary also created')
    print("Created vocabulary of " + str(len(vocab["word2id"])) + " words.")

    print ('>> writing vocab data into vocab.pkl\n')
    with open('vocab.pkl','wb') as fp:
        pickle.dump(vocab,fp)


def sen2enco(sentence):  #ak - this encodes the sentence into list of word indices
  return [getID(word, create=False) for word in nltk.word_tokenize(sentence)[:MAX_INPUT_LENGTH]]  


if __name__ == '__main__':
    # if sys.argv[1] == '--create_data':
        create_data_set_and_load_into_pickle()
    # elif sys.argv[1] == '--generate_vocab':
        generate_vocab()    
