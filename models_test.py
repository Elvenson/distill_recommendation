import numpy as np
import tensorflow as tf

from models import TwoTowerModel

BATCH_SIZE = 5


def _get_dummy_config():
    """
    Generate dummy config to initialize model.
    :return: A dictionary contains model config.
    """
    return dict(
        item_emb_shape=(10, 16),
        item_gender_emb_shape=(3, 4),
        item_age_emb_shape=(3, 4),
        nlayers=[32, 16]
    )


def _get_dummy_data():
    """
    Generate dummy data for testing model.
    :return: A dictionary contains training data.
    """
    query_itemid = np.random.randint(low=0, high=10, size=BATCH_SIZE)
    candidate_itemid = np.random.randint(low=0, high=10, size=BATCH_SIZE)
    query_item_gender = np.random.randint(low=0, high=3, size=BATCH_SIZE)
    candidate_item_gender = np.random.randint(low=0, high=3, size=BATCH_SIZE)
    query_item_age = np.random.randint(low=0, high=3, size=BATCH_SIZE)
    candidate_item_age = np.random.randint(low=0, high=3, size=BATCH_SIZE)

    return dict(
        query_itemid=query_itemid,
        query_item_age=query_item_age,
        query_item_gender=query_item_gender,
        candidate_itemid=candidate_itemid,
        candidate_item_age=candidate_item_age,
        candidate_item_gender=candidate_item_gender
    )


class ModelTest(tf.test.TestCase):
    def test_two_tower_model(self):
        data = _get_dummy_data()
        config = _get_dummy_config()
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

        model = TwoTowerModel(config)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.1))
        model_history = model.fit(
            dataset,
            validation_data=dataset,  # Just for testing, for real data we should use different dataset.
            validation_freq=1,
            epochs=1,
            verbose=1
        )

        self.assertIsNotNone(model_history, "Failed to train two-tower model.")
        self.assertTrue(
            len(model_history.history) != 0,
            'Empty two-tower model training log.'
        )


if __name__ == "__main__":
    tf.test.main()
