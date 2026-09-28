"""Click-prediction models (TensorFlow/Keras).

The three architectures are the ones in the original Colab code
(codes/matrix_factorization_recommender.py, codes/hybrid_ncf_model.py,
codes/enhanced_hybrid_ncf_model.py). Layer sizes and regularisation follow the
original files; only the input plumbing is shared so that all three models are
trained and evaluated by the same loop. Index 0 of the user table is reserved
for users that do not occur in the training period.
"""
import tensorflow as tf
from tensorflow.keras import regularizers


def _tower(sizes, l2_reg, dropout):
    layers = []
    for s in sizes:
        layers += [tf.keras.layers.Dense(s, activation="relu",
                                         kernel_regularizer=regularizers.l2(l2_reg)),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.Dropout(dropout)]
    return tf.keras.Sequential(layers)


class MatrixFactorization(tf.keras.Model):
    """MF with user/item biases (matrix_factorization_recommender.py)."""

    def __init__(self, num_users, num_items, embedding_size=32, l2_reg=1e-6, **_):
        super().__init__()
        init = "glorot_normal"
        reg = regularizers.l2(l2_reg)
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer=init, embeddings_regularizer=reg)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size, embeddings_initializer=init, embeddings_regularizer=reg)
        self.user_bias = tf.keras.layers.Embedding(num_users, 1, embeddings_initializer="zeros", embeddings_regularizer=reg)
        self.item_bias = tf.keras.layers.Embedding(num_items, 1, embeddings_initializer="zeros", embeddings_regularizer=reg)
        self.global_bias = tf.Variable(0.0, trainable=True)

    def call(self, x, training=False):
        u = self.user_embedding(x["user_id"])
        i = self.item_embedding(x["news_id"])
        dot = tf.reduce_sum(u * i, axis=1, keepdims=True)
        return tf.nn.sigmoid(dot + self.user_bias(x["user_id"]) + self.item_bias(x["news_id"]) + self.global_bias)


class HybridNCF(tf.keras.Model):
    """GMF branch + title-feature tower -> MLP head (hybrid_ncf_model.py)."""

    def __init__(self, num_users, num_items, embedding_size=32, l2_reg=1e-6, dropout_rate=0.3, **_):
        super().__init__()
        reg = regularizers.l2(l2_reg)
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_regularizer=reg)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size, embeddings_regularizer=reg)
        self.title_dense = _tower([128, 64], l2_reg, dropout_rate)
        self.mlp = _tower([64, 32], l2_reg, dropout_rate)
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x, training=False):
        gmf = self.user_embedding(x["user_id"]) * self.item_embedding(x["news_id"])
        t = self.title_dense(x["title_embedding"], training=training)
        return self.out(self.mlp(tf.concat([gmf, t], -1), training=training))


class EnhancedHybridNCF(tf.keras.Model):
    """GMF + title + abstract towers + category/subcategory embeddings -> MLP
    (enhanced_hybrid_ncf_model.py, April 2026 version)."""

    def __init__(self, num_users, num_items, num_categories, num_subcategories, embedding_size=32,
                 l2_reg=1e-6, dropout_rate=0.3, **_):
        super().__init__()
        reg = regularizers.l2(l2_reg)
        init = "glorot_normal"
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer=init, embeddings_regularizer=reg)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size, embeddings_initializer=init, embeddings_regularizer=reg)
        self.category_embedding = tf.keras.layers.Embedding(num_categories, 16, embeddings_initializer=init, embeddings_regularizer=reg)
        self.subcategory_embedding = tf.keras.layers.Embedding(num_subcategories, 16, embeddings_initializer=init, embeddings_regularizer=reg)
        self.title_dense = _tower([128, 64], l2_reg, dropout_rate)
        self.abstract_dense = _tower([128, 64], l2_reg, dropout_rate)
        self.mlp = _tower([128, 64, 32], l2_reg, dropout_rate)
        self.out = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)

    def call(self, x, training=False):
        gmf = self.user_embedding(x["user_id"]) * self.item_embedding(x["news_id"])
        t = self.title_dense(tf.nn.l2_normalize(x["title_embedding"], -1), training=training)
        a = self.abstract_dense(tf.nn.l2_normalize(x["abstract_embedding"], -1), training=training)
        h = tf.concat([gmf, t, a, self.category_embedding(x["category_id"]),
                       self.subcategory_embedding(x["subcategory_id"])], -1)
        return self.out(self.mlp(h, training=training))


MODELS = {"MF": MatrixFactorization, "H-NCF": HybridNCF, "EH-NCF": EnhancedHybridNCF}
