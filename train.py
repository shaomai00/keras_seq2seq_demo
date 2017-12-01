from keras.layers import Embedding, Activation, RepeatVector,TimeDistributed,Dense
from keras.models import Sequential
import time
import seq2seq
from seq2seq.models import SimpleSeq2Seq,AttentionSeq2Seq
import numpy as np
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from keras.preprocessing import sequence


src_max_len = 1 #输入的最大长度
trg_max_len = 1 #输出的最大长度
embedding_vector_dim = 20

if __name__ == '__main__':
    source, target = [], [] #保存原始的文本 形如：[['b', 's', 'a', 'q', 'q'], ['n', 'p', 'y'], ['l', 'b', 'w', 'u', 'j']....
    vocab = set({})

    with open('data/letters_source.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            src_sent = [character for character in line.strip()]
            source.append(src_sent)
            vocab.update(src_sent)
            src_max_len = src_max_len if src_max_len > len(src_sent) else len(src_sent)
    with open('data/letters_target.txt', 'r', encoding='utf-8', errors='ignore') as f1:
        for line in f1:
            trg_sent = [character for character in line.strip()]
            target.append(trg_sent)
            vocab.update(trg_sent)
            trg_max_len = trg_max_len if trg_max_len > len(trg_sent) else len(trg_sent)

    # the first words is the padding sign:
    vocablist = ['<s>'] + list(vocab)

    vocab_size = len(vocablist) #词表大小

    word_to_idx = dict((c, i) for i, c in enumerate(vocablist))  # 编码时需要将字符映射成数字index [在生成词表时已经添加标识符，此处不用再+1]
    idx_to_word = dict((i, c) for i, c in enumerate(vocablist))  # 解码时需要将数字index映射成字符
    source_sents =[ [word_to_idx[w] for w in sent] for sent in source]
    target_sents =[ [word_to_idx[w] for w in sent] for sent in target]

    trg_max_len = trg_max_len + 1
    print('src_max_len:',src_max_len,'trg_max_len:', trg_max_len)
    from keras.preprocessing import sequence

    source_pp = sequence.pad_sequences(source_sents, maxlen=src_max_len) #在前面padding了
    target_pp = sequence.pad_sequences(sequence.pad_sequences(target_sents, maxlen=trg_max_len - 1), maxlen=trg_max_len) #错一位padding
    trg_end_padding = np.pad(target_pp[:, 1:], [(0, 0), (0, 1)], 'constant', constant_values=0)
    print(trg_end_padding[0])

    print("vocab size:", vocab_size)
    inputs_train = source_pp
    print('source shape:', source_pp.shape)
    tars_train = np.eye(vocab_size)[trg_end_padding]
    print('target shape:', tars_train.shape)

    en_de_model = Sequential()
    en_de_model.add(Embedding(vocab_size, embedding_vector_dim, input_length=src_max_len)) #这里可以很方便的替换成预训练的embedding look-up层
    en_de_model.add(
        AttentionSeq2Seq(input_length=src_max_len, input_dim=embedding_vector_dim, hidden_dim=embedding_vector_dim,
                         output_length=trg_max_len, output_dim=embedding_vector_dim))
    en_de_model.add(TimeDistributed(Dense(vocab_size, activation='relu')))
    en_de_model.add(Activation('softmax'))
    en_de_model.summary()
    en_de_model.compile(loss='categorical_crossentropy', optimizer='adam')

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(inputs_train, tars_train, test_size=0.1,
                                                          random_state=1000)
    class PredictValid(Callback):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            super(PredictValid, self).__init__()
        def on_epoch_end(self, epoch, logs={}):
            out_predicts = en_de_model.predict(self.x)
            for i_idx, out_predict in enumerate(self.y):
                predict_sequence = []
                for predict_vector in out_predict:
                    next_index = np.argmax(predict_vector)
                    next_token = idx_to_word[next_index]
                    predict_sequence.append(next_token)
                print('Target output:', predict_sequence)
            for i_idx, out_predict in enumerate(out_predicts):
                predict_sequence = []
                for predict_vector in out_predict:
                    next_index = np.argmax(predict_vector)
                    next_token = idx_to_word[next_index]
                    predict_sequence.append(next_token)
                print('Predict output:', predict_sequence)

    predictValid = PredictValid(X_valid[:2], y_valid[:2])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')

    callbacks_list = [predictValid,early_stopping]
    en_de_model.fit(
        X_train, y_train,
        batch_size=50,
        epochs=100,
        validation_data=(X_valid, y_valid),
        verbose=2,
        callbacks=callbacks_list
    )