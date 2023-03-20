import tensorflow as tf


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
    For more info, we can refer to this paper: https://arxiv.org/pdf/1706.03762.pdf
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
    For more info, we can refer to this paper https://arxiv.org/pdf/2012.08986.pdf
    """
    def __init__(
            self,
            emb_size=32,
            num_buckets=20,
            temperature=0.0001,
            alpha=0.1,
            leaky_alpha=0.3,
            seed=42
    ):
        """
        Auto discretize layer
        :param emb_size: Final embedding dimension size.
        :param num_buckets: Number of buckets for aggregation.
        :param temperature: Temperature value for logit outputs.
        :param alpha: Control factor of skip-connection.
        :param leaky_alpha: Alpha param for leaky activation.
        :param seed: A seed number of weight initialization.
        """
        super(AutoDis, self).__init__()
        self.num_buckets = num_buckets
        self.emb_size = emb_size
        self.temperature = temperature
        self.alpha = alpha

        self.meta_emb = self.add_weight("meta_emb",
                                        shape=[num_buckets, emb_size],
                                        trainable=True,
                                        initializer=tf.keras.initializers.GlorotNormal(seed=seed))  # num_buckets x dim
        self.fc1 = tf.keras.layers.Dense(num_buckets)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)
        self.fc2 = tf.keras.layers.Dense(num_buckets)

    def call(self, x_dense):
        """
        Return embedding for a numeric feature.
        :param x_dense: Numeric feature with shape (batch_size, 1)
        :return: Feature embedding with shape (batch_size, dimension)
        """
        logits = self.leaky_relu(self.fc1(x_dense))  # batch_size x num_buckets
        logits = self.fc2(logits) + self.alpha * logits  # batch_size x num_buckets
        output = tf.nn.softmax(logits / self.temperature, axis=-1)  # batch_size x num_buckets
        x_emb = tf.linalg.matmul(output, self.meta_emb)  # batch_size x emb_dim

        return x_emb
