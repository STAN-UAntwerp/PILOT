from __future__ import annotations

import uuid
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout


class tree(object):
    """
    We use a tree object to save the PILOT model.
    Attributes:
    -----------
    node: str,
        type of the regression model
        'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
    pivot: tuple,
        a tuple to indicate where we performed a split. The first
        coordinate is the feature_id and the second one is
        the pivot.
    lm_l: ndarray,
        a 1D array to indicate the linear model for the left child node. The first element
        is the coef and the second element is the intercept.
    lm_r: ndarray,
        a 1D array to indicate the linear model for the right child node. The first element
        is the coef and the second element is the intercept.
    Rt: float,
        a real number indicating the rss in the present node.
    depth: int,
        the depth of the current node/subtree
    interval: ndarray,
        1D float array for the range of the selected predictor in the training data
    pivot_c: ndarry,
        1D int array. Indicating the levels in the left node
        if the selected predictor is categorical
    """

    def __init__(
        self,
        node=None,
        pivot=None,
        lm_l=None,
        lm_r=None,
        Rt=None,
        depth=None,
        interval=None,
        pivot_c=None,
    ) -> None:
        """
        Here we input the tree attributes.
        parameters:
        ----------
        node: str,
            type of the regression model
            'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
        pivot: tuple,
            a tuple to indicate where we performed a split. The first
            coordinate is the feature_id and the second one is
            the pivot.
        lm_l: ndarray,
            a 1D array to indicate the linear model for the left child node. The first element
            is the coef and the second element is the intercept.
        lm_r: ndarray,
            a 1D array to indicate the linear model for the right child node. The first element
            is the coef and the second element is the intercept.
        Rt: float,
            a real number indicating the rss in the present node.
        depth: int,
            the depth of the current node/subtree
        interval: ndarray,
            1D float array for the range of the selected predictor in the training data
        pivot_c: ndarry,
            1D int array. Indicating the levels in the left node
            if the selected predictor is categorical
        """
        self.left = None  # go left by default if node is 'lin'
        self.right = None
        self.Rt = Rt
        self.node = node
        self.pivot = pivot
        self.lm_l = lm_l
        self.lm_r = lm_r
        self.depth = depth
        self.interval = interval
        self.pivot_c = pivot_c

    def nodes_selected(self, depth=None) -> dict[str, int]:
        """
        count the number of models selected in the tree
        parameters:
        -----------
        depth: int, default = None.
            If specified count the number of models until the
            specified depth.
        """
        empty_tree = {"con": 0, "lin": 0, "blin": 0, "pcon": 0, "plin": 0, "pconc": 0}
        if self.node == "END":
            return empty_tree
        elif depth is not None and self.depth == depth + 1:
            # find the first node that reaches depth + 1
            return empty_tree
        nodes_l = self.left.nodes_selected(depth) if self.left is not None else empty_tree
        nodes_l[self.node] += 1
        if self.node in ["plin", "pcon"]:
            nodes_r = self.right.nodes_selected(depth) if self.right is not None else empty_tree
            for k in nodes_l.keys():
                nodes_l[k] += nodes_r[k]
        return nodes_l

    @staticmethod
    def get_depth(model_tree):
        depth = model_tree.depth
        left = model_tree.left
        right = model_tree.right
        if left is not None and left.node != "END":
            depth = tree.get_depth(left)
        if right is not None and right.node != "END":
            depth = max(depth, tree.get_depth(right))
        return depth


def _get_child_data(training_data, model_tree):
    left_data = training_data.copy()
    right_data = training_data.copy()
    if model_tree.node == "pconc":
        left_data = training_data[
            np.isin(training_data[:, model_tree.pivot[0]], model_tree.pivot_c)
        ]
        right_data = training_data[
            ~np.isin(training_data[:, model_tree.pivot[0]], model_tree.pivot_c)
        ]

    elif model_tree.node in ["plin", "pcon", "blin"]:
        left_data = training_data[training_data[:, model_tree.pivot[0]] <= model_tree.pivot[1]]
        right_data = training_data[training_data[:, model_tree.pivot[0]] > model_tree.pivot[1]]

    return left_data, right_data


def construct_graph(
    model_tree: tree, G: nx.Graph, training_data: np.ndarray, current_id: str = None
):
    current_node = model_tree.node
    depth = model_tree.depth
    if current_id is None:
        current_id = "0"
        G.add_node(
            f"{current_node}_{current_id}_{depth}",
            depth=depth,
            pivot=model_tree.pivot,
            lm_l=model_tree.lm_l,
            lm_r=model_tree.lm_r,
            interval=model_tree.interval,
            pivot_c=model_tree.pivot_c,
            samples_remaining=len(training_data),
            n_unique=len(training_data),
        )
    left_data, right_data = _get_child_data(training_data, model_tree)
    if model_tree.left is not None:
        left_id = str(uuid.uuid4().fields[-1])[-6:]
        post_fix = f"_{model_tree.left.depth}" if model_tree.left.depth is not None else ""
        n_unique = (
            len(np.unique(left_data[:, model_tree.left.pivot[0]]))
            if model_tree.left.pivot is not None
            else len(left_data)
        )
        G.add_node(
            f"{model_tree.left.node}_{left_id}{post_fix}",
            depth=model_tree.left.depth,
            pivot=model_tree.left.pivot,
            lm_l=model_tree.left.lm_l,
            lm_r=model_tree.left.lm_r,
            interval=model_tree.left.interval,
            pivot_c=model_tree.left.pivot_c,
            samples_remaining=len(left_data),
            n_unique=n_unique,
        )
        G.add_edge(
            f"{current_node}_{current_id}_{depth}", f"{model_tree.left.node}_{left_id}{post_fix}"
        )
        G = construct_graph(model_tree.left, G, left_data, left_id)
    if model_tree.right is not None:
        right_id = str(uuid.uuid4().fields[-1])[-6:]
        post_fix = f"_{model_tree.right.depth}" if model_tree.right.depth is not None else ""
        n_unique = (
            len(np.unique(right_data[:, model_tree.right.pivot[0]]))
            if model_tree.right.pivot is not None
            else len(right_data)
        )
        G.add_node(
            f"{model_tree.right.node}_{right_id}{post_fix}",
            depth=model_tree.right.depth,
            pivot=model_tree.right.pivot,
            lm_l=model_tree.right.lm_l,
            lm_r=model_tree.right.lm_r,
            interval=model_tree.right.interval,
            pivot_c=model_tree.right.pivot_c,
            samples_remaining=len(right_data),
            n_unique=n_unique,
        )
        G.add_edge(
            f"{current_node}_{current_id}_{depth}", f"{model_tree.right.node}_{right_id}{post_fix}"
        )
        G = construct_graph(model_tree.right, G, right_data, right_id)
    return G


def visualize_tree(model_tree: tree, training_data: np.ndarray, figsize: tuple = (20, 10)):
    G = construct_graph(model_tree, nx.Graph(), training_data)

    plt.figure(figsize=figsize)
    pos = graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(
        G, pos, node_size=1000, node_color="white", edgecolors="black", linewidths=2, alpha=0.1
    )
    nx.draw_networkx_edges(G, pos)

    get_label = (
        lambda n: n[0].split("_")[0]
        + (("_" + n[0].split("_")[-1]) if len(n[0].split("_")) == 3 else "")
        + "\n"
        + str(n[1].get("samples_remaining", ""))
    )
    _ = (
        nx.draw_networkx_labels(
            G, pos, labels={n[0]: get_label(n) for n in G.nodes(data=True)}, font_size=12
        ),
    )
