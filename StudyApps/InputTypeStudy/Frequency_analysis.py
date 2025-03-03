# %% IMPORTS
from FileHandling import *
from AnalysisFunctions import *
from scipy import fftpack
from scipy.fft import fft, fftfreq
import seaborn as sns
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal

# %% File reading -BEFORE!
subjects = range(24)
cursorTypes = ['Head', 'Eye', 'Hand']
repetitions = range(1, 5)
postures = ["Stand", 'Treadmill', "Walk"]
summary = pd.DataFrame(
    columns=['subject', 'posture', 'cursor', 'selection', 'target', 'repetition', 
             'H_sample_freq', 
             'V_sample_freq', 
             'H_power', 
             'V_power', 
             'H_sig_fft',
             'V_sig_fft',
             'sig_fft_new',  'sig_H', 'sig_V',
             
             ])
for subject, repetition, cursor, posture in itertools.product(subjects, repetitions, cursorTypes, postures):

    try:
        f, s = read_data(subject, repetition, cursor, 'Dwell', posture)
        d = split_target(f)
        for t in range(9):
            trial_summary = {'subject': subject, 'posture': posture, 'cursor': cursor,
                             'selection': "Dwell", 'target': t, 'repetition': repetition}
            data = d[t]
            data.timestamp = data.timestamp - data.timestamp.values[0]
            data['success'] = data.cursor_angular_distance <= 3.0
            # temp_data['success'] = temp_data.target_name == "Target_" + str(t)
            only_success = data[(data.success == True) | (data.target_name == "Target_" + str(t))]
            if len(only_success) <= 0:
                raise ValueError('no success frames', len(only_success))

            initial_contact_time = only_success.timestamp.values[0]
            dwell = data[data.timestamp <= initial_contact_time]
            # before = data[data.timestamp <=initial_contact_time]
            if dwell.cursor_angular_distance.mean() >= 13.789776700316207:
                continue
            # if (dwell.timestamp.values[-1] - dwell.timestamp.values[0]) < 120 / 60:
            #     continue
            # else:
            #     dwell = dwell[-120:].copy()
            # dwell.horizontal_offset =dwell.horizontal_offset - dwell.horizontal_offset.mean()
            # sig_H = signal.detrend(dwell.horizontal_offset.values)
            # sig_V = signal.detrend(dwell.vertical_offset.values)
            sig_H = dwell.horizontal_offset.values
            sig_V = dwell.vertical_offset.values
            trial_summary['sig_H'] = sig_H
            trial_summary['sig_V'] = sig_V
            # sig = dwell.horizontal_offset.values
            time_step = 1 / 60

            n=256
            fs=60
            H_fft_result = fftpack.fft(sig_H,n)
            H_fft_result = np.array(H_fft_result)
            H_psd_trials = (np.abs(H_fft_result) ** 2) / (n * fs)
            
            H_psd_trials = H_psd_trials[:n//2]
            
            H_freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
            trial_summary['H_sample_freq'] =H_freqs
            trial_summary['H_sig_fft'] = H_fft_result[:n//2]
            trial_summary['H_power'] = H_psd_trials
            
            V_fft_result = fftpack.fft(sig_V,n)
            V_fft_result = np.array(V_fft_result)
            V_psd_trials = (np.abs(V_fft_result) ** 2) / (n * fs)
            
            V_psd_trials = V_psd_trials[:n//2]
            
            V_freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
            trial_summary['V_sample_freq'] =V_freqs
            trial_summary['V_sig_fft'] = V_fft_result[:n//2]
            trial_summary['V_power'] = V_psd_trials
            # H_sig_fft = fftpack.fft(sig_H,256)
            # # # And the power (sig_fft is of complex dtype)
            # H_power = np.abs(H_sig_fft) ** 2
            # # # The corresponding frequencies
            # H_sample_freq = fftpack.fftfreq(256, d=time_step)
            # # # Find the peak frequency: we can focus on only the positive frequencies
            # H_pos_mask = np.where(H_sample_freq > 0)
            # freqs = H_sample_freq[H_pos_mask]
            # trial_summary['H_sample_freq'] = H_sample_freq[H_pos_mask]
            # trial_summary['H_sig_fft'] = H_sig_fft[H_pos_mask]
            # trial_summary['H_power'] = H_power[H_pos_mask]

            # V_sig_fft = fftpack.fft(sig_V,256)
            # # # And the power (sig_fft is of complex dtype)
            # V_power = np.abs(V_sig_fft) ** 2
            # # # The corresponding frequencies
            # V_sample_freq = fftpack.fftfreq(256, d=time_step)
            # # # Find the peak frequency: we can focus on only the positive frequencies
            # V_pos_mask = np.where(V_sample_freq > 0)
            # freqs = V_sample_freq[V_pos_mask]
            # trial_summary['V_sample_freq'] =V_sample_freq[V_pos_mask]
            # trial_summary['V_sig_fft'] = V_sig_fft[V_pos_mask]
            # trial_summary['V_power'] = V_power[V_pos_mask]
            
            summary.loc[len(summary)] = trial_summary
            # break
    except Exception as e:
        print(e, subject, repetition, cursor, posture)
        # return None
summary.to_pickle("Frequencies_before.pkl")
# %% File reading
subjects = range(24)
cursorTypes = ['Head', 'Eye', 'Hand']
repetitions = range(1, 5)
postures = ["Stand", 'Treadmill', "Walk"]
summary = pd.DataFrame(
    columns=['subject', 'posture', 'cursor', 'selection', 'target', 'repetition', 
             'H_sample_freq', 
             'V_sample_freq', 
             'H_power', 
             'V_power', 
             'H_sig_fft',
             'V_sig_fft',
             'sig_fft_new',  'sig_H', 'sig_V',
             
             ])
for subject, repetition, cursor, posture in itertools.product(subjects, repetitions, cursorTypes, postures):

    try:
        f, s = read_data(subject, repetition, cursor, 'Dwell', posture)
        d = split_target(f)
        for t in range(9):
            trial_summary = {'subject': subject, 'posture': posture, 'cursor': cursor,
                             'selection': "Dwell", 'target': t, 'repetition': repetition}
            data = d[t]
            data.timestamp = data.timestamp - data.timestamp.values[0]
            data['success'] = data.cursor_angular_distance <= 3.0
            # temp_data['success'] = temp_data.target_name == "Target_" + str(t)
            only_success = data[(data.success == True) | (data.target_name == "Target_" + str(t))]
            if len(only_success) <= 0:
                raise ValueError('no success frames', len(only_success))

            initial_contact_time = only_success.timestamp.values[0]
            dwell = data[data.timestamp > initial_contact_time]
            # before = data[data.timestamp <=initial_contact_time]
            if dwell.cursor_angular_distance.mean() >= 13.789776700316207:
                continue
            # if (dwell.timestamp.values[-1] - dwell.timestamp.values[0]) < 120 / 60:
            #     continue
            # else:
            #     dwell = dwell[-120:].copy()
            # dwell.horizontal_offset =dwell.horizontal_offset - dwell.horizontal_offset.mean()
            # sig_H = signal.detrend(dwell.horizontal_offset.values)
            # sig_V = signal.detrend(dwell.vertical_offset.values)
            sig_H = dwell.horizontal_offset.values
            sig_V = dwell.vertical_offset.values
            trial_summary['sig_H'] = sig_H
            trial_summary['sig_V'] = sig_V
            # sig = dwell.horizontal_offset.values
            time_step = 1 / 60

            n=256
            fs=60
            H_fft_result = fftpack.fft(sig_H,n)
            H_fft_result = np.array(H_fft_result)
            H_psd_trials = (np.abs(H_fft_result) ** 2) / (n * fs)
            
            H_psd_trials = H_psd_trials[:n//2]
            
            H_freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
            trial_summary['H_sample_freq'] =H_freqs
            trial_summary['H_sig_fft'] = H_fft_result[:n//2]
            trial_summary['H_power'] = H_psd_trials
            
            V_fft_result = fftpack.fft(sig_V,n)
            V_fft_result = np.array(V_fft_result)
            V_psd_trials = (np.abs(V_fft_result) ** 2) / (n * fs)
            
            V_psd_trials = V_psd_trials[:n//2]
            
            V_freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
            trial_summary['V_sample_freq'] =V_freqs
            trial_summary['V_sig_fft'] = V_fft_result[:n//2]
            trial_summary['V_power'] = V_psd_trials
            # H_sig_fft = fftpack.fft(sig_H,256)
            # # # And the power (sig_fft is of complex dtype)
            # H_power = np.abs(H_sig_fft) ** 2
            # # # The corresponding frequencies
            # H_sample_freq = fftpack.fftfreq(256, d=time_step)
            # # # Find the peak frequency: we can focus on only the positive frequencies
            # H_pos_mask = np.where(H_sample_freq > 0)
            # freqs = H_sample_freq[H_pos_mask]
            # trial_summary['H_sample_freq'] = H_sample_freq[H_pos_mask]
            # trial_summary['H_sig_fft'] = H_sig_fft[H_pos_mask]
            # trial_summary['H_power'] = H_power[H_pos_mask]

            # V_sig_fft = fftpack.fft(sig_V,256)
            # # # And the power (sig_fft is of complex dtype)
            # V_power = np.abs(V_sig_fft) ** 2
            # # # The corresponding frequencies
            # V_sample_freq = fftpack.fftfreq(256, d=time_step)
            # # # Find the peak frequency: we can focus on only the positive frequencies
            # V_pos_mask = np.where(V_sample_freq > 0)
            # freqs = V_sample_freq[V_pos_mask]
            # trial_summary['V_sample_freq'] =V_sample_freq[V_pos_mask]
            # trial_summary['V_sig_fft'] = V_sig_fft[V_pos_mask]
            # trial_summary['V_power'] = V_power[V_pos_mask]
            
            summary.loc[len(summary)] = trial_summary
            # break
    except Exception as e:
        print(e, subject, repetition, cursor, posture)
        # return None
summary.to_pickle("Frequencies.pkl")

#%%KS test
from scipy.stats import ks_2samp
before_summary = pd.read_pickle("Frequencies_before.pkl")
summary = pd.read_pickle("Frequencies.pkl")
data=summary.copy()
before_data=before_summary.copy()
def nor(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
for col in ["H_power", "V_power"]:
    FFT_summmary = pd.DataFrame(columns=['posture', 'cursor', 'sample_freq', 'power', 'sig_fft','fft'])
    for condition, dataframe in data.groupby([data.posture, data.cursor]):
        d = dataframe.copy()
        trial_summary = {'posture': condition[0], 'cursor': condition[1],
                            }
        # plt.bar(d.H_sample_freq.mean(),d.H_power.mean())
        # plt.title(str(condition))
        # plt.show()
        
        trial_summary['sample_freq'] = d.H_sample_freq.mean()
        trial_summary['power'] = d[col].mean()
        trial_summary['fft']=d[col[0]+"_sig_fft"].mean()
        FFT_summmary.loc[len(FFT_summmary)] = trial_summary
    before_FFT_summmary = pd.DataFrame(columns=['posture', 'cursor', 'sample_freq', 'power', 'sig_fft','fft'])
    for condition, dataframe in before_data.groupby([before_data.posture, before_data.cursor]):
        d = dataframe.copy()
        trial_summary = {'posture': condition[0], 'cursor': condition[1],
                            }
        # plt.bar(d.H_sample_freq.mean(),d.H_power.mean())
        # plt.title(str(condition))
        # plt.show()
        
        trial_summary['sample_freq'] = d.H_sample_freq.mean()
        trial_summary['power'] = d[col].mean()
        trial_summary['fft']=d[col[0]+"_sig_fft"].mean()
        before_FFT_summmary.loc[len(before_FFT_summmary)] = trial_summary

    # for pos,cur in itertools.product(["Stand",'Treadmill','Walk'],["Eye",'Head','Hand']):
    # for jj,j in enumerate([0,2,1]):  # for cursor
    #     for ii,i in enumerate([1,2,0]):  # for posture
    #         # before = before_FFT_summmary[(before_FFT_summmary.posture==pos) & (before_FFT_summmary.cursor==cur)]
    #         # after = FFT_summmary[(FFT_summmary.posture==pos) & (FFT_summmary.cursor==cur)]
            
    #         # psd_1 = before['power']
    #         # psd_2 = after['power']
    #         # print(psd_1);print(psd_2)
    #         psd_1_db = 10 * np.log10(before_FFT_summmary.iloc[3 * i + jj].power)
    #         psd_2_db = 10 * np.log10(FFT_summmary.iloc[3 * i + jj].power)
    #         ks_statistic, p_value = ks_2samp(psd_1_db, psd_2_db)

    #         # Print the results
    #         print(f"\n{FFT_summmary.iloc[3 * i + jj].posture} : {FFT_summmary.iloc[3 * i + jj].cursor}")
    #         print(f'KS Statistic: {ks_statistic}')
    #         print(f'P-value: {p_value}')
    n_rows = FFT_summmary.shape[0]
    pairs = itertools.combinations(range(n_rows), 2)
    for (i, j) in pairs:
        
        row1 = nor(np.log10(FFT_summmary.iloc[i].power))
        row2 = nor(np.log10(FFT_summmary.iloc[j].power))
        ks_statistic, p_value = ks_2samp(row1, row2)
        if p_value >0.05 and FFT_summmary.iloc[i].cursor==FFT_summmary.iloc[j].cursor:
            print(f"{col}\n{FFT_summmary.iloc[i].posture} : {FFT_summmary.iloc[i].cursor} and {FFT_summmary.iloc[j].posture} : {FFT_summmary.iloc[j].cursor} ")
            print(f'KS Statistic: {ks_statistic}')
            print(f'P-value: {p_value}')
            # results.append({
            #     'Row1': i,
            #     'Row2': j,
            #     'KS Statistic': ks_statistic,
            #     'P-value': p_value
            # })
        
#%%
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(context='paper',  # 매체: paper, talk, poster
              style='whitegrid',  # 기본 내장 테마
              # palette='deep',       # 그래프 색
              font_scale=2,  # 글꼴 크기
              rc=custom_params)  # 그래프 세부 사항
summary = pd.read_pickle("Frequencies.pkl")
# summary = pd.read_pickle("Frequencies_before.pkl")
data = summary.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
for col in ["H_power", "V_power"]:
    FFT_summmary = pd.DataFrame(columns=['posture', 'cursor', 'sample_freq', 'power', 'sig_fft','fft'])
    for condition, dataframe in data.groupby([data.posture, data.cursor]):
        d = dataframe.copy()
        trial_summary = {'posture': condition[0], 'cursor': condition[1],
                            }
        # plt.bar(d.H_sample_freq.mean(),d.H_power.mean())
        # plt.title(str(condition))
        # plt.show()
        
        trial_summary['sample_freq'] = d.H_sample_freq.mean()
        trial_summary['power'] = d[col].mean()
        trial_summary['fft']=d[col[0]+"_sig_fft"].mean()
        FFT_summmary.loc[len(FFT_summmary)] = trial_summary
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    # colors = ['red', 'blue', 'green']
    colors = sns.color_palette('pastel').as_hex()

    for jj,j in enumerate([0,2,1]):  # for cursor
        for ii,i in enumerate([1,2,0]):  # for posture

            low_freq_max = 10
            freqs=FFT_summmary.iloc[3 * i + jj].sample_freq
            psd_db=10*np.log10(FFT_summmary.iloc[3 * i + jj].power)
            # psd_db = FFT_summmary.iloc[3 * i + jj].fft
            ax[j].plot(FFT_summmary.iloc[3 * i + jj].sample_freq, psd_db,
                        #    alpha=1. - 0.3 * i,
                        #    alpha=0.7,
                           label=str(FFT_summmary.iloc[3 * i + jj].posture))
            
            low_freq_indices = np.where(freqs <= low_freq_max)[0]

        ax[j].set_title(FFT_summmary.iloc[3 * i + jj].cursor)
        ax[j].grid(True)
        if jj==2:
            ax[j].set_xlabel("Frequency (Hz)")
        if jj==0:
            ax[j].set_ylabel('Power/Frequency (dB/Hz)')
        
            # ax[j,i].set_xlim(-1,40)
            # ax[j, i].set_xticks([0, 5, 10, 15, 20, 25, 30, 25, 40])
    # plt.show()
    if col == "H_power":
        fig.suptitle("Power Spectral Density (PSD) of Horizontal Axis")
    if col == "V_power":
        fig.suptitle("Power Spectral Density (PSD) of Vertical Axis")
    
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power/Frequency (dB/Hz)')
    # plt.grid(True)
    plt.tight_layout()
    # plt.legend()
    # plt.show()
    plt.savefig(col + "fft.png")
    # plt.savefig(col + "fft_before.png")

# %%
for col in ["sig_H", "sig_V"]:
    summary = pd.read_pickle("Frequencies.pkl")
    data = summary.copy()
    FFT_summmary = pd.DataFrame(columns=['posture', 'cursor', 'sample_freq', 'power', 'sig_fft'])
    for condition, dataframe in data.groupby([data.posture, data.cursor]):
        time_step = 1 / 60
        trial_summary = {'posture': condition[0], 'cursor': condition[1],
                         }
        d = dataframe.copy()
        H = np.concatenate(d[col].values)
        sig_fft = fftpack.fft(H)
        # And the power (sig_fft is of complex dtype)
        power = np.abs(sig_fft) ** 2
        # The corresponding frequencies
        sample_freq = fftpack.fftfreq(H.size, d=time_step)
        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        trial_summary['sample_freq'] = sample_freq[pos_mask]
        trial_summary['sig_fft'] = sig_fft[pos_mask]
        trial_summary['power'] = power[pos_mask]
        # plt.figure()
        # plt.plot(freqs,power[pos_mask] )
        # plt.title(str(condition))
        # plt.savefig(str(condition)+'.png')
        FFT_summmary.loc[len(FFT_summmary)] = trial_summary
    

#%%
    fig, ax = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=False)
    # colors = ['red', 'blue', 'green']
    colors = sns.color_palette('pastel').as_hex()

    for jj,j in enumerate([0,2,1]):  # for cursor
        for i in range(3):  # for posture
            ax[j].plot(FFT_summmary.iloc[3 * i + jj].sample_freq, np.abs(FFT_summmary.iloc[3 * i + jj].sig_fft),
                           alpha=1. - 0.3 * i,
                           label=str(FFT_summmary.iloc[3 * i + jj].posture))
                        # color=colors[i], alpha=1. - 0.3 * i)
            # sns.barplot(x=freqs, y=np.abs(fft_results.iloc[3 * i + j].sig_fft_new), ax=ax[j, i])
        # ax[j].set_title(FFT_summmary.iloc[3 * i + j].posture + " : " + FFT_summmary.iloc[3 * i + j].cursor)
        ax[j].set_title(FFT_summmary.iloc[3 * i + jj].cursor)
        if jj==2:
            ax[j].set_xlabel("Frequency (Hz)")
        if jj==0:
            ax[j].set_ylabel("power")
            # ax[j,i].set_xlim(-1,40)
            # ax[j, i].set_xticks([0, 5, 10, 15, 20, 25, 30, 25, 40])
    # plt.show()
    if col == "sig_H":
        fig.suptitle("FFT analysis on horizontal axis")
    if col == "sig_V":
        fig.suptitle("FFT analysis on vertical axis")
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(col + "fft.png")
# %%
data = summary.copy()
fft_results = data.groupby([data.posture, data.cursor]).sig_fft.mean().reset_index()
# fft_results = data.groupby([data.posture, data.cursor]).sig_fft_new.mean().reset_index()
fft_results_sample_freq = data.groupby([data.posture, data.cursor]).sample_freq.mean().reset_index()
# fft_results_sample_freq = data.groupby([data.posture, data.cursor]).sample_freq_new.mean().reset_index()
freqs = fft_results_sample_freq.iloc[0].sample_freq
# freqs = fft_results_sample_freq.iloc[0].sample_freq_new
fft_results.insert(3, "freqs", None, allow_duplicates=False)
fft_results['freqs'] = [freqs, freqs, freqs, freqs, freqs, freqs, freqs, freqs, freqs]

# sns.catplot(data=fft_results,x='freqs',y='sig_fft',row='posture',col='cursor')
fig, ax = plt.subplots(3, 3, figsize=(16, 8), sharex=True, sharey=False)
for i in range(3):  # for posture
    for j in range(3):  # for cursor
        sns.barplot(x=freqs, y=np.abs(fft_results.iloc[3 * i + j].sig_fft), ax=ax[j, i])
        # sns.barplot(x=freqs, y=np.abs(fft_results.iloc[3 * i + j].sig_fft_new), ax=ax[j, i])
        ax[j, i].set_title(fft_results.iloc[3 * i + j].posture + " : " + fft_results.iloc[3 * i + j].cursor)
        # ax[j,i].set_xlim(-1,40)
        ax[j, i].set_xticks([0, 5, 10, 15, 20, 25, 30, 25, 40])
plt.savefig("fft.jpg")
# plt.show()
# %%
g = sns.catplot(
    data=summary, kind="box",
    x="posture", y="peak_freq", hue="cursor",
    # errorbar="sd",
    palette="dark", showfliers=False,
)
plt.show()
g = sns.catplot(
    data=summary, kind="box",
    x="posture", y="second_freq", hue="cursor",
    # errorbar="sd",
    palette="dark", showfliers=False,
)
plt.show()
g = sns.catplot(
    data=summary, kind="box",
    x="posture", y="third_freq", hue="cursor",
    # errorbar="sd",
    palette="dark", showfliers=False,
)
plt.show()

#%% Kolmogorov–Smirnov test
from scipy import stats
# data1=stats.zscore(FFT_summmary.iloc[0].sig_fft)
# data2=stats.zscore(FFT_summmary.iloc[1].sig_fft)
for i in range(len(FFT_summmary)):
    for j in range(i + 1, len(FFT_summmary)):
        data1=stats.zscore(FFT_summmary.iloc[i].power)
        data2=stats.zscore(FFT_summmary.iloc[j].power)
        ks_statistic, p_value = stats.ks_2samp(data1, data2)

        # Print the test statistic and p-value
        print(f"Comparison between row {i+1} and row {j+1}:")
        print("Kolmogorov-Smirnov statistic:", ks_statistic)
        print("p-value:", p_value)
        print()
# %%
