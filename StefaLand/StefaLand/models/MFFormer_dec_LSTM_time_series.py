import torch
from torch import nn
from MFFormer.layers.mask import MaskGenerator
from MFFormer.layers.positional_encoding import PositionalEncoding
from MFFormer.layers.transformer_layers import TransformerBackbone
from MFFormer.layers.features_embedding import TimeSeriesEncEmbedding, TimeSeriesDecEmbedding

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        embed_dim = configs.d_model
        self.embed_dim = embed_dim
        d_ffd = configs.d_ffd

        self.time_series_variables = configs.time_series_variables
        self.static_variables = configs.static_variables

        # # norm the layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # time series embedding
        self.time_series_embedding = TimeSeriesEncEmbedding(configs.time_series_variables, embed_dim,
                                                            dropout=configs.dropout)
        self.static_embedding = nn.Linear(len(self.static_variables), embed_dim)

        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=configs.dropout)

        # define the mask function
        self.mask_generator = MaskGenerator(configs.mask_ratio_time_series, mask_ratio_static=configs.mask_ratio_static,
                                            min_window_size=configs.min_window_size,
                                            max_window_size=configs.max_window_size)

        # define the encoder
        self.encoder = TransformerBackbone(embed_dim, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout)

        # define the decoder
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

        # define the projection layer
        self.time_series_projection = TimeSeriesDecEmbedding(configs.time_series_variables, embed_dim,
                                                             dropout=configs.dropout,
                                                             add_input_noise=configs.add_input_noise)
        self.dropout = nn.Dropout(configs.dropout)
        self.init_weights()

    def init_weights(self):

        def init_layer(layer):
            # nn.init.kaiming_uniform_(layer.weight, a=0.02)
            nn.init.uniform_(layer.weight, -self.configs.init_weight, self.configs.init_weight)
            nn.init.uniform_(layer.bias, -self.configs.init_bias, self.configs.init_bias)

        nn.init.uniform_(self.positional_encoding.position_embedding, -self.configs.init_weight, self.configs.init_weight)

        layers_to_init = [
            *self.time_series_embedding.embeddings1.values(),
            *self.time_series_embedding.embeddings2.values(),
            # *self.static_embedding,
            *self.time_series_projection.embeddings1.values(),
            *self.time_series_projection.embeddings2.values(),
        ]

        for layer in layers_to_init:
            init_layer(layer)

        nn.init.uniform_(self.time_series_embedding.masked_values, -self.configs.init_weight, self.configs.init_weight)
        nn.init.uniform_(self.static_embedding.weight, -self.configs.init_weight, self.configs.init_weight)

    def forward(self, batch_data_dict, is_mask=True):

        batch_x = batch_data_dict['batch_x']  # [batch_size, seq_len, num_features]
        batch_c = batch_data_dict['batch_c']  # [batch_size, num_features]
        masked_time_series_index = batch_data_dict['batch_time_series_mask_index']
        masked_static_index = batch_data_dict['batch_static_mask_index']

        # generate the mask
        if is_mask:
            if masked_time_series_index.numel() == 0:
                _, masked_time_series_index = self.mask_generator(batch_x.shape, method="consecutive")
                # _, masked_static_index = self.mask_generator(batch_c.shape, method="isolated_point")

            masked_time_series_index = masked_time_series_index.to(batch_x.device)
            # masked_static_index = masked_static_index.to(batch_x.device)

            # missing data mask
            masked_missing_time_series_index = torch.isnan(batch_x)

            # replace the missing value with 0
            batch_x = batch_x.masked_fill(masked_missing_time_series_index, 0)

            # combine the mask
            masked_time_series_index = masked_time_series_index | masked_missing_time_series_index

        else:
            unmasked_time_series_index, masked_time_series_index = None, None

        # replace the masked value with 0
        batch_x = batch_x.masked_fill(masked_time_series_index, 0)
        batch_c = batch_c.masked_fill(torch.isnan(batch_c), 0)

        # features embedding, [batch_size, seq_len, d_model]
        enc_x = self.time_series_embedding(batch_x, feature_order=self.time_series_variables,
                                           masked_index=masked_time_series_index)
        enc_c = self.static_embedding(batch_c)  # [batch_size, d_model]

        enc_x = enc_x + enc_c[:, None, :]

        # add the positional encoding
        enc_x = self.positional_encoding(enc_x)

        # warm-up the encoder
        enc_bs, enc_seq_len, enc_d_model = enc_x.shape
        if self.configs.warmup_train:
            enc_x = torch.cat([enc_x[:, :int(enc_x.shape[1] / 2), :], enc_x], dim=1)

        # encoder
        hidden_states = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)

        if self.configs.num_dec_layers > 0:
            # decoder
            dec_x, _ = self.decoder(hidden_states)
            dec_x = self.decoder_norm(dec_x)
            dec_x = self.dropout(dec_x)
        else:
            dec_x = hidden_states

        # remove the warm-up
        dec_x = dec_x[:, -enc_seq_len:, :]

        dec_x_time_series = dec_x  # [batch_size, seq_len, d_model]

        # restore
        outputs_time_series = self.time_series_projection(dec_x_time_series, feature_order=self.time_series_variables)
        
        output_dict = {
            'outputs_time_series': outputs_time_series,
            'outputs_static': None,
            'masked_time_series_index': masked_time_series_index,
            'masked_static_index': masked_static_index,
            # 'unmasked_time_series_index': unmasked_time_series_index.to(outputs_time_series.device),
            # 'unmasked_static_index': unmasked_static_index.to(outputs_time_series.device),
            "masked_missing_time_series_index": masked_missing_time_series_index,
            "masked_missing_static_index": None,
            'static_variables_dec_index_start': None,
            'static_variables_dec_index_end': None

        }

        return output_dict
