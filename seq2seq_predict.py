# coding = utf-8
import numpy as np
import pickle
from tensorflow import keras

class SeqModel:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 100
        self.latent_dim = 256
        self.num_samples = 10000
        self.data_path = "fra.txt"
        self.model_path = "model/s2s_model.h5"
        self.input_index_path = "model/input_vocab.pkl"
        self.target_index_path = "model/output_vocab.pkl"
        self.config_path = "model/config.txt"
        self.config_dict = self.load_config()
        self.encoder_model, self.decoder_model, self.input_token_index, self.target_token_index, self.reverse_input_char_index, self.reverse_target_char_index = self.load_model()
        self.num_decoder_tokens = self.config_dict["Number of unique output tokens"]
        self.max_decoder_seq_length = self.config_dict["Max sequence length for outputs"]
        self.num_encoder_tokens = self.config_dict["Number of unique input tokens"]
        self.max_encoder_seq_length = self.config_dict["Max sequence length for inputs"]
        return

    """加载预训练词表"""
    def load_vocab(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    """加载配置文件"""
    def load_config(self):
        config_dict = {}
        with open(self.config_path, 'r') as f:
            for line in f:
                line = line.strip().split(':')
                if not line:
                    continue
                config_dict[line[0]] = int(line[1])
        return config_dict

    """对测试数据进行编码"""
    def encode_data(self, input_texts):
        encoder_input_data = np.zeros((len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32")
        for i, input_text in enumerate(input_texts):
            for t, char in enumerate(input_text):
                if char not in self.input_token_index:
                    self.input_token_index[char] = self.input_token_index["unk"]
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
        return encoder_input_data


    """加载模型，并存入模型结构"""
    def load_model(self):
        input_token_index = self.load_vocab(self.input_index_path)
        target_token_index = self.load_vocab(self.target_index_path)
        model = keras.models.load_model(self.model_path)
        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = keras.Model(encoder_inputs, encoder_states)
        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(self.latent_dim,), name="input_3")
        decoder_state_input_c = keras.Input(shape=(self.latent_dim,), name="input_4")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )
        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

        return encoder_model, decoder_model, input_token_index, target_token_index, reverse_input_char_index, reverse_target_char_index

    """对序列进行解码"""
    def decode_sequence(self, len_sent, input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, num_decoder_tokens):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index["\t"]] = 1.0
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            if sampled_char == "\n" or len(decoded_sentence) > len_sent+1:
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0
            states_value = [h, c]
        return decoded_sentence

    """利用模型进行生成预测"""
    def predict(self, sent):
        input_seq = self.encode_data([sent])
        return self.decode_sequence(len(sent), input_seq, self.encoder_model, self.decoder_model, self.target_token_index,
                                    self.reverse_target_char_index, self.num_decoder_tokens)

if __name__ == '__main__':
    handler = SeqModel()
    while 1:
        sent = input('enter an sent to trans:').strip()
        res = handler.predict(sent)
        print(res)