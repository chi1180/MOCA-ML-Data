import pandas as pd
from sentence_transformers import SentenceTransformer

""" config """
MODEL_NAME = "intfloat/multilingual-e5-small"
model = SentenceTransformer(MODEL_NAME)


def generate_vector(text):
    return model.encode(text)
