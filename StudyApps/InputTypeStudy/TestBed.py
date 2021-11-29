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
# for s in range(24):
#     # collect_offsets(s)
#     summarize_subject(s)

# summary = visualize_summary(show_plot=True)
summary_dataframe = visualize_offsets(show_plot=False)

# %%
head_data = summary_dataframe[(summary_dataframe['posture'] == 'WALK') & (summary_dataframe['cursor_type'] == 'HEAD')]
head_horizontals = []
for hh in head_data.horizontal.values:
    head_horizontals += list(hh)
parameters = {}
for ct, pos, ax in itertools.product(['EYE', 'HAND', 'HEAD'], ['WALK', 'STAND'], ['horizontal', 'vertical']):
    head_data = summary_dataframe[(summary_dataframe['posture'] == pos) & (summary_dataframe['cursor_type'] == ct)]
    head_horizontals = []
    for hh in head_data[ax].values:
        head_horizontals += list(hh)
    import seaborn as sns

    sns.displot(head_horizontals, kde=True)
    head_horizontals = np.array(head_horizontals)
    sigma = head_horizontals.std()
    mean = head_horizontals.mean()
    # plt.axvline(mean)
    # plt.axvline(mean+2*sigma)
    # plt.axvline(mean-2*sigma)
    # plt.axvline(mean+3*sigma)
    # plt.axvline(mean-3*sigma)
    # plt.title(str(ct)+'_'+str(pos)+'_'+str(ax))

    # plt.show()
    print(ct, pos, ax, mean, sigma, 3 * sigma)
    parameters[(ct, pos, ax)] = sigma
    """
    EYE WALK horizontal 0.24722741899600512 4.420237751534142 13.260713254602425
    EYE WALK vertical -0.07572955993240958 2.4375580926867078 7.312674278060124
    EYE STAND horizontal 0.20836864617779696 1.5635038623192548 4.690511586957764
    EYE STAND vertical 0.18935830234235748 1.491778058469321 4.475334175407963
    HAND WALK horizontal -0.09866966003016135 6.521336309396893 19.564008928190677
    HAND WALK vertical -0.26136047800482975 1.6178699940290733 4.85360998208722
    HAND STAND horizontal 0.04440054874312414 1.2868251691549768 3.8604755074649306
    HAND STAND vertical 0.05941293945469808 1.3437840646867873 4.0313521940603625
    HEAD WALK horizontal -0.0040063012538855474 5.0511439371221885 15.153431811366566
    HEAD WALK vertical 0.3109466494847721 2.3182985184738376 6.954895555421513
    HEAD STAND horizontal -0.0398596984122698 1.303755389483091 3.9112661684492736
    HEAD STAND vertical 0.1434393747167268 1.5906082672928836 4.771824801878651
    """
# %%

# data = read_hololens_data(0, 'WALK', 'HEAD', t,False)
without_stand = summarize_subject(0, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], True, False)
with_stand = summarize_subject(23, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], False, False)
without_walk = summarize_subject(1, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], True, False)
with_walk = summarize_subject(23, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], False, False)
# %%
print('stand,accuracy', with_stand.mean_offset.mean(), '->', without_stand.mean_offset.mean())
print('walk,accuracy', with_walk.mean_offset.mean(), '->', without_walk.mean_offset.mean())
print('stand,precision', with_stand.std_offset.mean(), '->', without_stand.std_offset.mean())
print('walk,precision', with_walk.std_offset.mean(), '->', without_walk.std_offset.mean())
# %%
import seaborn as sns

colorset = ['maroon', 'orangered', 'darkorange', 'olive', 'yellowgreen', 'darkolivegreen', 'turquoise', 'deepskyblue',
            'dodgerblue']
postures = ['STAND','WALK']
# postures = ['WALK']
sigma_multiple = 3
for posture, cursor_type in itertools.product(postures, ['EYE', 'HAND', 'HEAD']):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    size_fig,size_ax = plt.subplots(1,1,figsize=(10,10))
    for t in range(9):
        for w in ['SMALL', 'LARGE']:
            if w == 'LARGE':
                wide = 14.04
            else:
                wide = 7.125
            # wide = 10
            x_offset = wide * math.sin(t * math.pi / 9 * 2)
            y_offset = wide * math.cos(t * math.pi / 9 * 2)
            h_raw = summary_dataframe.loc[
                (summary_dataframe['posture'] == posture) & (summary_dataframe['wide'] == w) & (
                        summary_dataframe['cursor_type'] == cursor_type) & (
                        summary_dataframe['target_num'] == t)]['horizontal'].values[0]
            v_raw = summary_dataframe.loc[
                (summary_dataframe['posture'] == posture) & (summary_dataframe['wide'] == w) & (
                        summary_dataframe['cursor_type'] == cursor_type) & (
                        summary_dataframe['target_num'] == t)]['vertical'].values[0]
            h = []
            v = []
            for i in range(len(h_raw)):
                if not (-sigma_multiple * sigmas[(cursor_type, posture, 'horizontal')] < h_raw[i] < sigma_multiple *
                        sigmas[
                            (cursor_type, posture, 'horizontal')]):
                    continue
                elif not (-sigma_multiple * sigmas[(cursor_type, posture, 'vertical')] < v_raw[i] < sigma_multiple *
                          sigmas[
                              (cursor_type, posture, 'vertical')]):
                    continue
                h.append(h_raw[i])
                v.append(v_raw[i])
            h = np.array(h)
            v = np.array(v)
            # sns.kdeplot(x=h + x_offset, y=v + y_offset, fill=True, ax=ax)
            ax.scatter(x_offset,y_offset,s=100,c=colorset[t],marker='x')
            # ax.scatter(h + x_offset, v + y_offset, s=0.5, alpha=0.05, c=colorset[t])
            plt_confidence_ellipse(h + x_offset, v + y_offset, ax, 2, edgecolor=colorset[t], linestyle='--',linewidth=3)
            size_ax.scatter(x_offset,y_offset,s=100,c=colorset[t],marker='x')
            import matplotlib.patches as patches
            width = h.mean()+2*h.std()
            height = v.mean() + 2*v.std()
            size_ax.add_patch(
                patches.Rectangle(
                    (x_offset - width,y_offset-height)
                    ,2*width,2*height,edgecolor=colorset[t],fill=False
                )
            )
    plt.title(str(posture) + "," + str(cursor_type) + '-' + str(sigma_multiple))
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
