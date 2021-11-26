# %%
import itertools
import math

import numpy as np

import pandas as pd
from collections import defaultdict
from FileHandling import *

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from AnalysisFunctions import *

# from scipy.spatial.transform import Rotation as R

pio.renderers.default = 'browser'
pd.set_option('mode.chained_assignment', None)  # <==== 경고를 끈다

# %%
t = 5
# data = get_one_trial(0, 'WALK', 'HEAD', 5, 0)

# data = read_hololens_data(0, 'WALK', 'HEAD', t,True)
for s in range(24):
    # collect_offsets(s)
    summarize_subject(s)

summary = visualize_summary(show_plot=True)
# summary_datafame = visualize_offsets(show_plot=False)
# %%
import seaborn as sns
colorset = ['maroon','orangered','darkorange','olive','yellowgreen','darkolivegreen','turquoise','deepskyblue','dodgerblue']
for posture, cursor_type in itertools.product(['STAND', 'WALK'], ['EYE', 'HAND', 'HEAD']):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for t in range(9):
        for w in ['SMALL', 'LARGE']:
            if w == 'LARGE':
                wide = 14.04
            else:
                wide = 7.125
            # wide = 10
            x_offset = wide * math.sin(t * math.pi / 9 * 2)
            y_offset = wide * math.cos(t * math.pi / 9 * 2)
            h = summary_datafame.loc[
                (summary_datafame['posture'] == posture) & (summary_datafame['wide'] == w) & (
                            summary_datafame['cursor_type'] == cursor_type) & (
                        summary_datafame['target_num'] == t)]['horizontal'].values[0]
            v = summary_datafame.loc[
                (summary_datafame['posture'] == posture) & (summary_datafame['wide'] == w) & (
                            summary_datafame['cursor_type'] == cursor_type) & (
                        summary_datafame['target_num'] == t)]['vertical'].values[0]
            # sns.kdeplot(x=h + x_offset, y=v + y_offset, fill=True, ax=ax)
            ax.scatter(h+x_offset,v+y_offset,s=0.5,alpha=0.1,c=colorset[t])
            plt_confidence_ellipse(h+x_offset,v+y_offset,ax,2,edgecolor=colorset[t],linestyle=':')
    plt.title(str(posture) + "," + str(cursor_type))
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.show()
# print('accuracy',hs.mean(),vs.mean())
# print('precision',hs.std(),vs.std())

# %%confidence ellipse
summary = visualize_summary(show_plot=True)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# summary = summary[summary.error.isna() == True]
ax.scatter(np.array(summary['mean_offset_horizontal']), np.array(summary['mean_offset_vertical']), s=3)

confidence_ellipse(np.array(summary['mean_offset_horizontal']), np.array(summary['mean_offset_vertical']), ax,
                   edgecolor='red', n_std=3,
                   linestyle='--', label='overall')
confidence_ellipse(np.array(summary['mean_offset_horizontal']), np.array(summary['mean_offset_vertical']), ax,
                   edgecolor='red', n_std=2,
                   linestyle='--', label='overall')
# ax.scatter(summary['mean_offset_horizontal'].mean(), summary['mean_offset_vertical'].mean(), marker='x', s=100,
#            color='magenta')
ax.set_title("overall")
ax.legend()
fig.show()

# %%
# import numpy as np
#
#
# def confidence_ellipse(x, y, n_std=1.96, size=100):
#     """
#     Get the covariance confidence ellipse of *x* and *y*.
#     Parameters
#     ----------
#     x, y : array-like, shape (n, )
#         Input data.
#     n_std : float
#         The number of standard deviations to determine the ellipse's radiuses.
#     size : int
#         Number of points defining the ellipse
#     Returns
#     -------
#     String containing an SVG path for the ellipse
#
#     References (H/T)
#     ----------------
#     https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
#     https://community.plotly.com/t/arc-shape-with-path/7205/5
#     """
#     if x.size != y.size:
#         raise ValueError("x and y must be the same size")
#
#     cov = np.cov(x, y)
#     pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
#     # Using a special case to obtain the eigenvalues of this
#     # two-dimensionl dataset.
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#     theta = np.linspace(0, 2 * np.pi, size)
#     ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])
#
#     # Calculating the stdandard deviation of x from
#     # the squareroot of the variance and multiplying
#     # with the given number of standard deviations.
#     x_scale = np.sqrt(cov[0, 0]) * n_std
#     x_mean = np.mean(x)
#
#     # calculating the stdandard deviation of y ...
#     y_scale = np.sqrt(cov[1, 1]) * n_std
#     y_mean = np.mean(y)
#
#     translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
#     rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
#                                 [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
#     scale_matrix = np.array([[x_scale, 0],
#                              [0, y_scale]])
#     ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix
#
#     path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
#     for k in range(1, len(ellipse_coords)):
#         path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
#     path += ' Z'
#     return path
#
#
# """ EXAMPLE """
# from plotly import graph_objects as go
# from plotly.colors import DEFAULT_PLOTLY_COLORS
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
#
# iris = load_iris()
#
# pca = PCA(n_components=2)
# scores = pca.fit_transform(iris.data)
#
# fig = go.Figure()
#
# for target_value, target_name in enumerate(iris.target_names):
#     color = DEFAULT_PLOTLY_COLORS[target_value]
#     fig.add_trace(
#         go.Scatter(
#             x=scores[iris.target == target_value, 0],
#             y=scores[iris.target == target_value, 1],
#             name=target_name,
#             mode='markers',
#             marker={'color': color}
#         )
#     )
#
#     fig.add_shape(type='path',
#                   path=confidence_ellipse(scores[iris.target == target_value, 0],
#                                           scores[iris.target == target_value, 1]),
#                   line={'dash': 'dot'},
#                   line_color=color)
#
# fig.show()
