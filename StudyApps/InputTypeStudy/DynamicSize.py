# %%
from FileHandling import *
from AnalysisFunctions import *
import seaborn as sns

# %%
# rates = np.arange(0.1, 2.1, 0.1).round(1)
# rates = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
rates=[1.0,1.1,1.2]
results = []
width=3
height=6
for rate in rates:
    d = target_expansion_simulation_subject(range(24), rate,width,height)
    results.append(d)

    simulation_summary = pd.concat(results)
    simulation_summary.to_csv("expansion" + str(rate)  +str(width)+str(height)+".csv")

# %%
rates = [2.0]
rates = np.arange(0.1, 2.1, 0.1).round(1)

# rates=[1.0,1.1,1.2]
width=6
height=3

data = []
for rate in rates:
    d = pd.read_csv("expansion" + str(rate) +str(width)+str(height)+ ".csv")
    # d = pd.read_csv("expansion" + str(rate) + ".csv")
    data.append(d)
data = pd.concat(data)
# data.drop(['selection'], axis=1, inplace=True)
data.success.replace(
    {"True": True, "False": False, "nan": None, "0.0": False, "1.0": True}, inplace=True
)
summary = (
    data.groupby([data.subject, data.rate, data.cursor, data.posture, data.selection])
    .success.mean()
    .reset_index()
)
# data['error_rate'] = data.apply(
#         lambda x: x['error_rate']+0.05 if x['selection']=='Dwell' and x['posture']== "Treadmill" else x['error_rate'], axis=1)
summary.loc[summary.posture == "Walk", "posture"] = "Circuit"
summary.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
summary["error_rate"] = (1 - summary.success) * 100
summary = summary[summary.Mobility != "Stand"]
# summary.rate = summary.rate *10
by_subject = pd.read_csv("newstudy_BySubject.csv")
by_subject.loc[by_subject.posture == "Walk", "posture"] = "Circuit"
by_subject.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
by_subject = by_subject[by_subject.Mobility != "Stand"]
by_subject["error_rate"] = 100 - by_subject.success
d = by_subject[["subject", "Modality", "Mobility", "Trigger", "success", "error_rate"]]
# d["rate"] = -1
# summary = pd.concat([summary, d])
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    # font_scale=7,  # 글꼴 크기
    font_scale=3,
    rc=custom_params,
)  # 그래프 세부 사항


# import patchworklib as pw
# by_subject = pd.read_csv("newstudy_BySubject.csv")
# summary["error_rate"] = summary.apply(
#     lambda x: (
#         x["error_rate"] + 3 - x["rate"] * 3
#         if x["Trigger"] == "Dwell"
#         and x["Mobility"] == "Treadmill"
#         and x["Modality"] != "Hand"
#         and x["rate"] <= 1.0
#         else x["error_rate"]
#     ),
#     axis=1,
# )
# summary["error_rate"] = summary.apply(
#     lambda x: (
#         x["error_rate"] + 1
#         if x["Trigger"] == "Dwell"
#         and x["Mobility"] == "Treadmill"
#         and x["Modality"] == "Hand"
#         and x["rate"] <= 1.0
#         else x["error_rate"]
#     ),
#     axis=1,
# )

plotdf= summary.groupby([summary.rate,summary.Modality,summary.Mobility,summary.Trigger]).mean().reset_index()
fig = px.line(plotdf, x='rate', y='error_rate', color='Modality',   facet_row='Trigger',   facet_col='Mobility',
             markers=True,category_orders={'Mobility': ['Treadmill','Circuit'], "Modality": ['Eye','Head','Hand']},
                 labels={'error_rate': 'Error Rate', 'rate': 'Expansion Rate'},
                 title='error rates by rate, Posture, and Cursor',

                #  symbol='Modality'
                 )

    # Customize the layout
fig.update_layout(
    title= 'Dynamic Target size: Error rates by rate, Posture, and Cursor',
    # xaxis_title='Window',
    # yaxis_title='Mean Measurement'
)
# fig.write_html(f"DynamicTargetSize"+str(width)+str(height)+".html", include_plotlyjs='cdn')
fig.show()
#%%
from plotly.subplots import make_subplots
plotdf_66 = pd.read_pickle("66dynamic.pkl");plotdf_66['size']="Full expansion"
plotdf_36 = pd.read_pickle("36dynamic.pkl");plotdf_36['size']="Vertical expansion"
plotdf_63 = pd.read_pickle("63dynamic.pkl");plotdf_63['size']="Horizontal expansion"
plotdf = pd.concat([plotdf_66,plotdf_63,plotdf_36],axis=1)
plotdf = pd.concat([plotdf_66,plotdf_63,plotdf_36],axis=0)
mod = 'Eye'
for mod in ['Eye','Head','Hand']:
    d = plotdf[plotdf.Modality==mod]
    fig1= px.line(d, x='rate', y='error_rate', color='size',   facet_row='Trigger',   facet_col='Mobility',
            markers=True,category_orders={'Mobility': ['Treadmill','Circuit'], "Modality": ['Eye','Head','Hand']},
                labels={'error_rate': 'Error Rate (%)', 'rate': 'Expansion Rate (°/s)'},symbol="size",
                title=mod,template='plotly_white',
                )
    fig1.update_yaxes(range=[0,60])
    # fig1.update_xaxes(range=[0,2])
    fig1.update_xaxes(    
        title_text="",row=1,col=2
    )
    fig1.update_layout(height=750,width=500)
    fig1.update_layout(showlegend=True)
    fig1.update_traces(marker=dict(size=5))
    colors=['Blue','Red',"Green"]
    if mod =='Eye':
        # Eye
        # Click,Treadmill
        row=2;col=1
        for i,n in enumerate([3.70,9.95,13.31]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Click, Circuit
        row=2;col=2
        for i,n in enumerate([2.20,12.15,20.25]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Dwell Treadmill
        row=1;col=1
        for i,n in enumerate([0.0,0.57,2.31]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Dwell Circuit
        row=1;col=2
        for i,n in enumerate([2.32,10.02,15.94]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
    elif mod=='Hand':
        # Click,Treadmill
        row=2;col=1
        for i,n in enumerate([4.62,10.18,13.98]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Click, Circuit
        row=2;col=2
        for i,n in enumerate([10.99,21.18,30.63]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Dwell Treadmill
        row=1;col=1
        for i,n in enumerate([2.26,5.14,17.36]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Dwell Circuit
        row=1;col=2
        for i,n in enumerate([10.65,25.65,45.89]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
    elif mod=="Head":
        # Click,Treadmill
        row=2;col=1
        for i,n in enumerate([6.71,12.73,13.17]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Click, Circuit
        row=2;col=2
        for i,n in enumerate([8.68,14.81,22.30]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Dwell Treadmill
        row=1;col=1
        for i,n in enumerate([0.46,1.96,3.58]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        # Dwell Circuit
        row=1;col=2
        for i,n in enumerate([10.87,16.50,36.01]):
            fig1.add_shape(type="line",x0=0, x1=2, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
    # fig1.show()
    
    fig1.write_image(str(mod)+"dynamic.png")
#%% Making plotdf files for every width and height

rates = np.arange(0.1, 2.1, 0.1).round(1)

# rates=[1.0,1.1,1.2]

width=6
height=6
data = []
for rate in rates:
    # d = pd.read_csv("expansion" + str(rate) +str(width)+str(height)+ ".csv")
    d = pd.read_csv("expansion" + str(rate) + ".csv")
    data.append(d)
data = pd.concat(data)
# data.drop(['selection'], axis=1, inplace=True)
data.success.replace(
    {"True": True, "False": False, "nan": None, "0.0": False, "1.0": True}, inplace=True
)
summary = (
    data.groupby([data.subject, data.rate, data.cursor, data.posture, data.selection])
    .success.mean()
    .reset_index()
)
# data['error_rate'] = data.apply(
#         lambda x: x['error_rate']+0.05 if x['selection']=='Dwell' and x['posture']== "Treadmill" else x['error_rate'], axis=1)
summary.loc[summary.posture == "Walk", "posture"] = "Circuit"
summary.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
summary["error_rate"] = (1 - summary.success) * 100
summary = summary[summary.Mobility != "Stand"]
plotdf= summary.groupby([summary.rate,summary.Modality,summary.Mobility,summary.Trigger]).mean().reset_index()

plotdf.to_pickle(str(width)+str(height)+'dynamic.pkl')
#%%
fig = sns.catplot(
    data=summary,
    x="rate",
    y="error_rate",
    hue="Modality",
    row="Trigger",
    col="Mobility",
    kind="bar",
    hue_order=["Eye", "Head", "Hand"],
    legend_out=False,
)
fig.fig.set_size_inches(30, 12)
axes = fig.axes.flatten()
axes[0].set_xticklabels(
    [
        "G",
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
    ]
)
axes[0].set_ylabel("Error Rate (%)")
axes[2].set_xlabel("Expansion Rate (deg/s)")
axes[2].set_ylabel("Error Rate (%)")
axes[3].set_xlabel("Expansion Rate (deg/s)")
# fig.refline(y=50, color='red')

# # if sel=='Click':
# axes[0].axvline(0.5,ls='--',linewidth=1,c='blue')
axes[0].set_ylim((0, 100))
# axes[0].set_ylim((0,100))
# axes[0].axhline(21.30,ls='--',linewidth=3,c='orange')
# axes[0].axhline(16.34,ls='--',linewidth=3,c='green')
# axes[1].axhline(28.52,ls='--',linewidth=3,c='blue')
# axes[1].axhline(30.91,ls='--',linewidth=3,c='orange')
# axes[1].axhline(32.32,ls='--',linewidth=3,c='green')
# #     elif sel=="Dwell":
# axes[2].axhline(5.90,ls='--',linewidth=3,c='blue')
# axes[2].axhline(9.23,ls='--',linewidth=3,c='orange')
# axes[2].axhline(20.64,ls='--',linewidth=3,c='green')
# axes[3].axhline(37.13,ls='--',linewidth=3,c='blue')
# axes[3].axhline(53.59,ls='--',linewidth=3,c='orange')
# axes[3].axhline(55.82,ls='--',linewidth=3,c='green')
plt.tight_layout()
# plt.show()
plt.savefig("DynamicTargetDimension.pdf")
