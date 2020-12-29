# MoviereviewsCNN
Classifying IMDB movie reviews using Convolutional Neural Networks

The data: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Model : Convolutional Neural Network

Embeddings: Word2vec model

# What are the process?
Clean the data by removing redundant words(stem words,email address,html tags)

Seperate each sentences into words using tokenizer

Lower the data

Lemmitize the words(Lemmitization is the process of identifying the base words. [[Caring’ -> Lemmatization -> ‘Care’,‘Caring’ -> Stemming -> ‘Car’]]

Seperate the data into positive reviews and negative reviews

## WordCloud

Use wordcloud to detect the most occurence word in both positive and negative reviews

## Indexing

Find ten positive words and negative words

## Number of grams 

Check number of bi grams and tri grams

## Split the data into training and testing set

Use scikit learn

## Store all the training and testing words in a list with its indexing

all_training_words = [i.split() for i in train_data['review']]

all_training_wordings = []

for i in range(len(all_training_words)):

    for j in range(len(all_training_words[i])):
    
        all_training_wordings.append(all_training_words[i][j])
        
training_vocab = sorted(list(set(all_training_wordings)))

## For test data

all_testing_words = [i.split() for i in test_data['review']]

all_testing_wordings = []

for i in range(len(all_testing_words)):

    for j in range(len(all_testing_words[i])):
    
        all_testing_wordings.append(all_testing_words[i][j])
        
testing_vocab = sorted(list(set(all_testing_wordings)))

## Tokenize the data by the number of vocabulary length

tokenizer_train = Tokenizer(num_words = len(training_vocab))

tokenizer_test = Tokenizer(num_words = len(testing_vocab))

## Fit the tokenizer in the X_train and X_test so that it will split the datas 

tokenizer_train.fit_on_texts(X_train)

tokenizer_test.fit_on_texts(X_test)

## Convert each word into the index it got assigned to

X_train = tokenizer_train.texts_to_sequences(X_train)

X_test = tokenizer_test.texts_to_sequences(X_test)

## Check the training vocabulary size

train_vocabulary_size = len(tokenizer_train.word_index)+1

test_vocabulary_size = len(tokenizer_test.word_index)+1

## Padding the training and testing data so that we have equal dimension when we are working with Neural Network

X_train = pad_sequences(X_train, padding = 'post',maxlen = MAX_SEQUENCE_LENGTH)

X_test =  pad_sequences(X_test,padding = 'post',maxlen = MAX_SEQUENCE_LENGTH)


## For this project I used google's word2vec to convert words into its own 300 dimension

Create an empty matrix with zeros the size of train_vocabulary and the embedding dimension so that we can create unique 300 embedding dimension for each word

## Finally train your Convolutional Neural Network

















