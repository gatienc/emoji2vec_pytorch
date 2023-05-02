# External dependencies
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import pickle as pk
import pandas as pd

# Internal dependencies
from model import Emoji2Vec,ModelParams

from parameter_parser import CliParser
from utils import build_kb, get_examples_from_kb, generate_embeddings, get_metrics, generate_predictions

import plotly.graph_objects as go

# Authorship
__author__ = "Gatien CHENU"
__email__ = "gatien.chenu@etu.unistra.fr"


def visualize(model):
    # setup
    args = CliParser()
    args.print_params('EMOJI VISUALIZATION')

    # mapping from emoji index to emoji
    mapping = pk.load(open('emoji_mapping_file.pkl', 'rb'))
    
    values = list(mapping.values())  # Utiliser les valeurs comme index
    # Créer un objet pandas.Index
    index_from_values = pd.Index(values)
    print(index_from_values)
    # get the embeddings
    V = model.nn.V.detach().numpy()
    print(V.shape)
    print(V)
    tsne = TSNE(perplexity=50, n_components=2, init='random', n_iter=300000, early_exaggeration=1.0,
                    n_iter_without_progress=1000)
    trans = tsne.fit_transform(V)
    x, y = zip(*trans)
    #plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="text",
            text=index_from_values,
            textposition="middle center",
            textfont=dict(size=18),  # Set the font size here
        )
    )
    fig.write_html("emoji2vec_visualization.html")
    # # plot the emoji using TSNE on matplotlib
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # plt.scatter(x, y, marker='o', alpha=0.0)

    # for i in range(len(trans)):
    #     ax.annotate(mapping[i], xy=trans[i], textcoords='data')

    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    params = {"data_folder": "data/training"}
    print('reading training data from: ' + params["data_folder"])
    kb, ind2phr, ind2emoji = build_kb(params["data_folder"])
    params.update({"mapping_file": "emoji_mapping_file.pkl"})
    params.update({"word2vec_embeddings_file": "data/word2vec/GoogleNews-vectors-negative300.bin.gz",
               "phrase_embeddings_file": "phrase_embeddings.pkl"})
    embeddings_array = generate_embeddings(ind2phr=ind2phr, kb=kb, embeddings_file=params["phrase_embeddings_file"],
                                    word2vec_file=params["word2vec_embeddings_file"])
    model_params = ModelParams(in_dim=300, 
                            out_dim=300, 
                            max_epochs=60, 
                            pos_ex=4, 
                            neg_ratio=1, 
                            learning_rate=0.001,
                            dropout=0.0, 
                            class_threshold=0.5)
    model = Emoji2Vec(model_params=model_params, num_emojis=kb.dim_size(0), embeddings_array=embeddings_array)
    model_folder = 'example_emoji2vec'
    model.nn = torch.load(model_folder + '/model.pt')

    visualize(model)