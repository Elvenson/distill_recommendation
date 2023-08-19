import tensorflow as tf
from keras.layers import Embedding
from tensorflow_recommenders.models import Model
from tensorflow_recommenders.tasks import Retrieval


class QueryModel(tf.keras.Model):
    """
    Query tower for two-tower model architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Some example features we can have.
        self.item_emb = Embedding(*config["item_emb_shape"])
        self.item_age_emb = Embedding(*config["item_age_emb_shape"])
        self.item_gender_emb = Embedding(*config["item_gender_emb_shape"])

        # Fully connected layer at the end to generate the final query embedding.
        self.dense_layers = tf.keras.Sequential()
        # Use activation for all but the last layer.
        for layer_size in config["nlayers"][:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(layer_size, activation="relu")
            )
            self.dense_layers.add(tf.keras.layers.Dropout(0.1))
        # No activation for the last layer.
        self.dense_layers.add(tf.keras.layers.Dense(config["nlayers"][-1]))

    def call(self, inputs, training: bool = False):
        item_emb = self.item_emb(inputs["query_itemid"])  # batch_size x dim1
        item_age_emb = self.item_age_emb(inputs["query_item_age"])  # batch_size x dim2
        item_gender_emb = self.item_gender_emb(inputs["query_item_gender"])  # batch_size x dim3

        # Concatenate all features and feed to final FC.
        fc_input = tf.concat([item_emb, item_age_emb, item_gender_emb], axis=-1)  # batch_size x dim4
        return self.dense_layers(fc_input)  # batch_size x final_dim


class CandidateModel(tf.keras.Model):
    """
    Candidate model for two-tower model architecture.
    """

    def __init__(self, config, query_model):
        super().__init__()
        self.config = config

        # Embedding layers are shared between query and candidate models.
        self.item_emb = query_model.item_emb
        self.item_age_emb = query_model.item_age_emb
        self.item_gender_emb = query_model.item_gender_emb

        # Fully connected layer at the end to generate the final candidate embedding.
        self.dense_layers = tf.keras.Sequential()
        # Use activation for all but the last layer.
        for layer_size in config["nlayers"][:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(layer_size, activation="relu")
            )
            self.dense_layers.add(tf.keras.layers.Dropout(0.1))
        # No activation for the last layer.
        self.dense_layers.add(tf.keras.layers.Dense(config["nlayers"][-1]))

    def call(self, inputs, training: bool = False):
        item_emb = self.item_emb(inputs["candidate_itemid"])  # batch_size x dim1
        item_age_emb = self.item_age_emb(inputs["candidate_item_age"])  # batch_size x dim2
        item_gender_emb = self.item_gender_emb(inputs["candidate_item_gender"])  # batch_size x dim3

        # Concatenate all features and feed to final FC.
        fc_input = tf.concat([item_emb, item_age_emb, item_gender_emb], axis=-1)  # batch_size x dim4
        return self.dense_layers(fc_input)  # batch_size x final_dim


class TwoTowerModel(Model):
    """
    A simple two-tower model for vector recall.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query_model = QueryModel(config)
        self.candidate_model = CandidateModel(config, self.query_model)

        # Use retrieval task to calculate in-batch softmax and categorical cross-entropy loss.
        self.task = Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            batch_metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(k=x, name="model1_top_{}_categorical_accuracy".format(x))
                for x in [1, 10, 100]],  # For calculating in-batch hit rate.
            num_hard_negatives=config.get("num_hard_negatives", None)
        )

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_emb = self.query_model(inputs)
        candidate_emb = self.candidate_model(inputs)

        loss = self.task(
            query_emb,
            candidate_emb
        )
        return loss
