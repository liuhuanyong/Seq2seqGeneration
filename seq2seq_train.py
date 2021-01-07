# coding = utf-8
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.utils import plot_model

class SeqModel:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 100
        self.latent_dim = 256
        self.num_samples = 400
        self.data_path = "data.txt"
        self.model_path = "model/s2s_model.h5"
        self.input_index_path = "model/input_vocab.pkl"
        self.target_index_path = "model/output_vocab.pkl"
        self.config_path = "model/config.txt"
        self.image_path = "image/lstm_seq2seq_model.png"
        return

    """存储预训练词表"""
    def write_vocab(self, path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return


    """存储配置文件"""
    def write_config(self, path, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length):
        with open(path, 'w+') as f:
            f.write("Number of unique input tokens:" + str(num_encoder_tokens) + '\n')
            f.write("Number of unique output tokens:" + str(num_decoder_tokens) + '\n')
            f.write("Max sequence length for inputs:" + str(max_encoder_seq_length) + '\n')
            f.write("Max sequence length for outputs:" + str(max_decoder_seq_length) + '\n')
        return

    """准备训练与测试数据"""
    def prepare_data(self):
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split("\t")
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        input_characters = sorted(list(input_characters)) + ["unk"]
        target_characters = sorted(list(target_characters)) + ["unk"]

        return input_texts, target_texts, input_characters, target_characters

    """将训练样本进行编码，编码成id-张量形式"""
    def encode_data(self, input_texts, target_texts, input_characters, target_characters):
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)

        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])
        self.write_config(self.config_path, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length)
        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


        self.write_vocab(self.input_index_path, input_token_index)
        self.write_vocab(self.target_index_path, target_token_index)
        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        print(input_texts)
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        return num_decoder_tokens, num_encoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data

    """搭建lstm编码与解码网络结构"""
    def build_model(self, num_decoder_tokens, num_encoder_tokens):
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
        encoder = keras.layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
        decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model

    """训练模型，打印模型参数，并绘制网络模型结构图"""
    def train_model(self):
        input_texts, target_texts, input_characters, target_characters = self.prepare_data()
        num_decoder_tokens, num_encoder_tokens,encoder_input_data, decoder_input_data, decoder_target_data = self.encode_data(input_texts, target_texts, input_characters, target_characters)
        model = self.build_model(num_decoder_tokens, num_encoder_tokens)
        model.summary()
        plot_model(model, to_file=self.image_path, show_shapes=True)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
        )
        model.save(self.model_path)
        return model

if __name__ == '__main__':
    handler = SeqModel()
    handler.train_model()