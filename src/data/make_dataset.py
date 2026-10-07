import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc=pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr=pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files=glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
#data_path="../../data/raw/MetaMotion/*.csv"
f=files[1]

participant=f.split("-")[0].split("\\")[1]
label=f.split("-")[1]
category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")


df=pd.read_csv(f)
df["participant"]=participant
df["label"]=label
df["category"]=category
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df=pd.DataFrame()
gyr_df=pd.DataFrame()

acc_set=1#just a convention to keep the group a number of rows say set 1 have 10 rows etc.
gyr_set=1

for f in files:
    # print(f)
    participant=f.split("-")[0].split("\\")[1]
    label=f.split("-")[1]
    category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df=pd.read_csv(f)
    
    df["participant"]=participant
    df["label"]=label
    df["category"]=category
    
    if "Accelerometer" in f:
        acc_df["set"]=acc_set
        acc_set+=1
        acc_df=pd.concat([acc_df,df])
    if "Gyroscope" in f:
        gyr_df["set"]=gyr_set
        gyr_set+=1
        gyr_df=pd.concat([gyr_df,df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()
pd.to_datetime(df["epoch (ms)"],unit="ms")
pd.to_datetime(df["time (01:00)"])# this will give you value 1 hr ahead of the actual time as the time is in UTC format and we are in IST format
df["time (01:00)"].dt.week# this will give error as can't use dt on the string object so we need to convert it to datetime first
pd.to_datetime(df["time (01:00)"]).dt.month


print(acc_df.index)
acc_df.index=pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")
print(acc_df.index)

del acc_df["epoch (ms)"]
del acc_df["elapsed (s)"]
del acc_df["time (01:00)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files=glob("../../data/raw/MetaMotion/*.csv")
len(files)

def read_data_from_files(files):
    
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()

    acc_set=1
    gyr_set=1

    for f in files:
        # print(f)
        participant=f.split("-")[0].split("\\")[1]
        label=f.split("-")[1]
        category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df=pd.read_csv(f)
        
        df["participant"]=participant
        df["label"]=label
        df["category"]=category
        
        if "Accelerometer" in f:
            df["set"]=acc_set
            acc_set+=1
            acc_df=pd.concat([acc_df,df])
        if "Gyroscope" in f:
            df["set"]=gyr_set
            gyr_set+=1
            gyr_df=pd.concat([gyr_df,df])
            
        
    acc_df.index=pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
    gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["elapsed (s)"]
    del acc_df["time (01:00)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df,gyr_df

acc_df,gyr_df=read_data_from_files(files)




# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

#axis=1 if want to stack horizontally and axis=0 if want to stack the data vertically

data_merged=pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1)
data_merged.head(50)
            
"""gyro scope measures data at very fast frequency hence there is so much 
diffrence in the number of data of both and dropna data
there was very small frequency at which there both frequency matched only 1119 number
of rows for both gyr and acc and we have to expand the time range so that we can get more number of datapoints conciding in both acc and gyr data"""

data_merged.columns =[
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
# resample means to combine data points to a different frequency

sampling ={
    "acc_x":"mean",
    "acc_y":"mean",
    "acc_z":"mean",
    "gyr_x":"mean",
    "gyr_y":"mean",
    "gyr_z":"mean",
    "participant":"last",
    "label":"last",
    "category":"last",
    "set":"last",
}
#200ms ki space me jo last valur hogi vo aa jayegi categorical columns me
data_merged[:1000].resample(rule="200ms").apply(sampling)
days=[g for n,g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled=pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"]=data_resampled["set"].astype("int64")
data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")