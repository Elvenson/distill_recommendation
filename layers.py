import tensorflow as tf
from keras.layers import Embedding


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer.
    """

    def __init__(self, dm, dk, dv, num_heads):
        """
        Multi-head attention implementation
        :param dm: Embedding dimension
        :param dk: Key dimension
        :param dv: Value dimension
        :param num_heads:
        """
        super(MultiHeadAttention, self).__init__()

        initializers = tf.keras.initializers.GlorotUniform()
        self.wk = tf.Variable(initializers(shape=(num_heads, dm, dk)), trainable=True)
        self.wq = tf.Variable(initializers(shape=(num_heads, dm, dk)), trainable=True)
        self.wv = tf.Variable(initializers(shape=(num_heads, dm, dv)), trainable=True)
        self.wo = tf.Variable(initializers(shape=(num_heads * dv, dm)), trainable=True)
        self.num_heads = num_heads
        self.model_dims = dm
        self.val_dim = dv

    @staticmethod
    def _scale_dot_product_attention(K, Q, V, mask=None):
        """
        Scale dot product in transformer
        :param K: Key embedding (..., key_sentence_length, key_dim)
        :param Q: Query embedding (..., query_sentence_length, key_dim)
        :param V: Value embedding (..., value_sentence_length, value_dim)
        :param mask: For masking future sentence.
        :return: Scaled value (..., query_sentence_length, value_dim)
        """
        qk_mat = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(K.shape[-1], tf.float32)
        scale_qk_mat = qk_mat / tf.math.sqrt(dk)

        if mask:
            scale_qk_mat -= (1 - mask) * 10 ** 9
        attention_weights = tf.nn.softmax(scale_qk_mat, axis=-1)

        v_mat = tf.matmul(attention_weights, V)
        return v_mat, attention_weights

    def call(self, q, k, v, mask=None):
        """
        Run multi head attention
        :param q: query input (batch_size, query_length, model_dim)
        :param k: key input (batch_size, key_length, model_dim)
        :param v: value input (batch_size, key_length, value_dim)
        :param mask: masking input
        :return: scaled value input with size (batch_size, query_length, model_dim),
        and attention score (query_length, key_length)
        """
        batch_size = q.shape[0]
        query_length = q.shape[-2]
        q = tf.expand_dims(q, 1)  # (batch_size, 1, query_length, model_dim)
        k = tf.expand_dims(k, 1)  # (batch_size, 1, query_length, model_dim)
        v = tf.expand_dims(v, 1)  # (batch_size, 1, query_length, value_dim)
        wk = tf.expand_dims(self.wk, 0)  # (1, num_heads, model_dim, dk)
        wq = tf.expand_dims(self.wq, 0)
        wv = tf.expand_dims(self.wv, 0)

        query_mat = tf.einsum('...ij,...jk->...ik', q, wq)  # (batch_size, num_heads, query_length, dk)
        key_mat = tf.einsum('...ij,...jk->...ik', k, wk)  # (batch_size, num_heads, key_length, dk)
        value_mat = tf.einsum('...ij,...jk->...ik', v, wv)  # (batch_size, num_heads, key_length, dv)

        scaled_val, attention = self._scale_dot_product_attention(key_mat, query_mat,
                                                                  value_mat,
                                                                  mask)  # (batch_size, num_heads, query_length, dv)
        scaled_val = tf.transpose(scaled_val, [0, 2, 1, 3])
        scaled_val = tf.reshape(scaled_val, [batch_size, query_length, self.val_dim * self.num_heads])
        output = tf.matmul(scaled_val, self.wo)

        return output, attention


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer in transformer architecture.
    For more info, we can refer to this paper: https://arxiv.org/pdf/1706.03762.pdf.
    """

    def __init__(self, dm, dk, dv, epsilon=1e-6, nhidden=32, num_head=8, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.dense1 = tf.keras.layers.Dense(nhidden, activation=tf.keras.layers.LeakyReLU())
        self.dense2 = tf.keras.layers.Dense(dm, activation=tf.keras.layers.LeakyReLU())
        self.attention_layer = MultiHeadAttention(dm, dk, dv, num_head)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=True, mask=None, output_attention=True):
        scaled_input, attention = self.attention_layer(inputs, inputs, inputs, mask)  # Self attention
        scaled_input = self.dropout1(scaled_input, training)
        inputs = self.layer_norm1(scaled_input + inputs)  # layer normalization + residual

        ffn = self.dropout2(self.dense2(self.dense1(inputs)), training)  # Fully connected layer
        inputs = self.layer_norm2(inputs + ffn)  # layer normalization + residual

        if output_attention:
            return inputs, attention
        return inputs


class AutoDis(tf.keras.layers.Layer):
    """
    Auto discretize layer for numeric features.
    For more info, we can refer to this paper https://arxiv.org/pdf/2012.08986.pdf.
    """

    def __init__(
            self,
            emb_size,
            autodis_num_bins=8,
            temperature=1,
            alpha=0.1
    ):
        """
        Auto discretize layer
        :param emb_size: Final embedding dimension size.
        :param autodis_num_bins: Number of bin for auto discretize layer.
        :param temperature: Temperature value for logit outputs.
        :param alpha: Control factor of skip-connection.
        """
        super(AutoDis, self).__init__()
        self.emb_size = emb_size
        self.temperature = temperature

        self.autodis_meta_emb = self.add_weight("autodis_meta_emb",  # num_bin x dim
                                                shape=[autodis_num_bins, emb_size],
                                                trainable=True)
        self.autodis_discrete = tf.keras.layers.Dense(autodis_num_bins)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.fc2 = tf.keras.layers.Dense(autodis_num_bins)
        self.alpha = alpha

    def build(self, input_shape):
        self.ranks = len(input_shape)

    def call(self, x_dense):
        """
        Return embedding for numeric feature.
        :param x_dense: Numeric features with shape (batch_size, 1) or shape (batch_size, timestamp, 1)
        :return: Feature embedding with shape (batch_size, dimension) or shape (batch_size, timestamp, dimension)
        """
        if self.ranks != 2 and self.ranks != 3:
            raise ValueError("Unsupported tensor with shape {}".format(x_dense.shape))
        if self.ranks == 2:
            x_dense = tf.expand_dims(x_dense, -1)
        autodis_logits = self.autodis_discrete(x_dense)
        autodis_logits = self.leaky_relu(autodis_logits)
        autodis_logits = self.fc2(autodis_logits) + self.alpha * autodis_logits
        autodis_logits = tf.nn.softmax(autodis_logits / self.temperature,
                                       axis=-1)  # batch_size x 1 or timesteps x num_bin
        x_emb = tf.einsum("bij, jk -> bik", autodis_logits, self.autodis_meta_emb)  # batch_size x dim

        if self.ranks == 2:
            x_emb = tf.squeeze(x_emb, axis=1)

        return x_emb


class BatchNorm(tf.keras.layers.Layer):
    """
    Batch Normalization layer implementation.
    For more detail, we can refer to this paper https://arxiv.org/pdf/1502.03167.pdf.
    """

    def __init__(self, epsilon=0.001, momentum=0.99):
        super(BatchNorm, self).__init__()
        self.epsilon = epsilon
        self.momentum = tf.constant([[momentum]], dtype=tf.float32)

    def build(self, input_shape):
        self.beta = self.add_weight(shape=(input_shape[1]), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(input_shape[1]), initializer='ones', trainable=True)
        self.moving_mean = self.add_weight(shape=(1, input_shape[1]), initializer='zeros', trainable=False)
        self.moving_var = self.add_weight(shape=(1, input_shape[1]), initializer='ones', trainable=False)

    def call(self, inputs, training=False):
        """
        Batch normalize layer.
        :param inputs: Input data with shape (batch_size, dim)
        :param training: A flag to know if this is training or inference.
        :return: Batch normalize output.
        """
        if training:
            mean, var = tf.nn.moments(inputs, axes=[0], keepdims=True)  # (1, dim)
            moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean  # (1, dim)
            moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var  # (1, dim)
            self.moving_mean.assign(moving_mean)
            self.moving_var.assign(moving_var)
            normalized_input = (inputs - mean) / tf.math.sqrt(var + self.epsilon)  # (batch_size, dim)

            return normalized_input * self.gamma + self.beta
        else:
            normalized_input = (inputs - self.moving_mean) / tf.math.sqrt(self.moving_var + self.epsilon)
            return normalized_input * self.gamma + self.beta


class LayerNorm(tf.keras.layers.Layer):
    """
    Layer normalization implementation.
    For more info, we can refer to this paper https://arxiv.org/pdf/1607.06450.pdf.
    """

    def __init__(self, epsilon=0.01):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = self.add_weight(shape=(input_shape[0], 1), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(input_shape[0], 1), initializer='ones', trainable=True)

    def call(self, inputs):
        """
        Layer norm layer.
        :param inputs: Input data with shape (batch_size, dim)
        :return: Layer normalize output.
        """
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)  # (batch_size, 1)
        normalized_input = (inputs - mean) / tf.math.sqrt(var + self.epsilon)  # (batch_size, dim)
        return normalized_input * self.gamma + self.beta


class CompositeEmbedding(tf.keras.layers.Layer):
    """
    Compositional embedding layer.
    For more info, we can refer to this paper https://arxiv.org/pdf/1909.02107.pdf.
    param nvoc: An integer represents the vocabulary size.
    param emb_dim: An integer represents embedding size.
    """

    def __init__(self, nvoc, emb_dim, *args, **kwargs):
        super(CompositeEmbedding, self).__init__(*args, **kwargs)
        self.nvoc = nvoc
        self.composite1 = Embedding(self.nvoc + 1, emb_dim, name="composite_1")
        self.composite2 = Embedding(self.nvoc + 1, emb_dim, name="composite_2")
        self.hashing_layer1 = tf.keras.layers.Hashing(num_bins=nvoc + 1, mask_value="0", salt=[6971, 7321])
        self.hashing_layer2 = tf.keras.layers.Hashing(num_bins=nvoc + 1, mask_value="0", salt=[7723, 7507])

    def build(self, input_shape):
        super(CompositeEmbedding, self).build(input_shape)

    def call(self, x):
        """
        Generate embedding based on tensor value.
        param x: A tensor of shape (batch_size, ).
        return An embedding tensor with shape (batch_size, emb_dim).
        """
        if x.dtype != tf.string:
            x = tf.as_string(x)

        idx1 = self.hashing_layer1(x)
        idx2 = self.hashing_layer2(x)

        x1 = self.composite1(idx1)
        x2 = self.composite2(idx2)
        emb = tf.einsum("...i,...i->...i", x1, x2)

        return emb


class SENet(tf.keras.layers.Layer):
    """
    Squeeze-Excitation network (SENET) architecture.
    For more info, we can refer to this paper: https://arxiv.org/pdf/1905.09433.pdf
    """

    def __init__(self, reduction_ratio):
        super(SENet, self).__init__()

        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        num_feats = input_shape[1]
        self.dense1 = tf.keras.layers.Dense(num_feats // self.reduction_ratio,
                                            activation=tf.keras.layers.ReLU(),
                                            use_bias=False,
                                            kernel_initializer=tf.initializers.he_normal)
        self.dense2 = tf.keras.layers.Dense(num_feats,
                                            activation=tf.keras.layers.ReLU(),
                                            use_bias=False,
                                            kernel_initializer=tf.initializers.he_normal)

    def call(self, inputs):
        """
        :param inputs: Input features with shape (batch_size, num_feature, dim).
        :return: Tensor with output (batch_size, num_feature, dim).
        """
        z = tf.reduce_mean(inputs, axis=-1)  # batch_size, num_feature
        w = self.dense2(self.dense1(z))  # batch_size, num_feature
        return inputs * tf.expand_dims(w, axis=-1)
