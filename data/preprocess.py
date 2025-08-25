#%%
import pandas as pd

admission = pd.read_csv('/Users/chen/Desktop/code/fed-ldp-quantile-reg/data/admissions.csv')
patient = pd.read_csv('/Users/chen/Desktop/code/fed-ldp-quantile-reg/data/patients.csv')
print(admission.shape)
print(patient.shape)

# merge
admission_patient = pd.merge(admission, patient, on='subject_id', how='left')
print(admission_patient.shape)

#%%
import matplotlib.pyplot as plt

# calc hospital stay and plot di｜stribution
admission_patient['hospital_stay'] = pd.to_datetime(admission_patient['dischtime']) - pd.to_datetime(admission_patient['admittime'])
admission_patient['hospital_stay_days'] = admission_patient['hospital_stay'].dt.days

# 过滤异常值（例如：住院天数为负或超过一年的记录），使分布更合理
hospital_stay_days = admission_patient['hospital_stay_days']
hospital_stay_days = hospital_stay_days[(hospital_stay_days >= 0) & (hospital_stay_days <= 365)]

# 绘制住院时长分布直方图
plt.figure(figsize=(10, 6))
plt.hist(hospital_stay_days, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
plt.xlabel('Hospital Stay (days)')
plt.ylabel('Frequency')
plt.title('Distribution of Hospital Stay')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # 自动调整布局，避免标签被截断
plt.show()

print("住院时长统计信息（天）：")
print(hospital_stay_days.describe())


# %%
admission_provider = admission_patient['admit_provider_id'].unique()
print('num of admission_provider:', len(admission_provider))
print(admission_patient['admit_provider_id'].describe())

# %%
