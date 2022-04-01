# %%
import pandas as pd
import os
import numpy as np

# %%
data_dir = os.path.join(os.getcwd(), "auc", "negative-selection",
                        "negative-selection", "syscalls")
cert_dir = os.path.join(data_dir, "snd-cert")
unm_dir = os.path.join(data_dir, "snd-unm")

cert_1_data = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.1.test"), names=["string"])
cert_1_labels = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.1.labels"), names=["label"])

cert_2_data = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.2.test"), names=["string"])
cert_2_labels = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.2.labels"), names=["label"])

cert_3_data = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.3.test"), names=["string"])
cert_3_labels = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.3.labels"), names=["label"])

unm_1_data = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.1.test"), names=["string"])
unm_1_labels = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.1.labels"), names=["label"])

unm_2_data = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.2.test"), names=["string"])
unm_2_labels = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.2.labels"), names=["label"])

unm_3_data = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.3.test"), names=["string"])
unm_3_labels = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.3.labels"), names=["label"])

# %%
# Processing Function


def process_data(dataset, labels, data_save_path, label_save_path, index_save_path):
    print(dataset.shape[0])
    min_length = 7
    label_vec = labels["label"]
    with open(data_save_path, "w") as data_file:
        with open(label_save_path, "w") as label_file:
            with open(index_save_path, "w") as index_file:
                for idx, row in enumerate(dataset["string"]):
                    if(idx % 100 == 0):
                        print(f"Processing Entry {idx}")
                    count = 0
                    letter_count = min_length
                    while letter_count <= len(row):
                        index_file.write(str(idx))
                        index_file.write("\n")
                        data_file.write(row[count:count+min_length])
                        data_file.write("\n")
                        label_file.write(str(label_vec[idx]))
                        label_file.write("\n")
                        count += 1
                        letter_count += 1


# %%
save_folder = os.path.join(os.getcwd(), "test")

# %%
# --------------------- Processing Dataset Unm 1 ------------------
process_data(unm_1_data, unm_1_labels, os.path.join(
    save_folder, "unm_1.test"), os.path.join(save_folder, "unm_1.labels"), os.path.join(save_folder, "unm_1.indices"))

# %%
# ---------------------- Processing Dataset Unm 2 --------------------
process_data(unm_2_data, unm_2_labels, os.path.join(
    save_folder, "unm_2.test"), os.path.join(save_folder, "unm_2.labels"), os.path.join(save_folder, "unm_2.indices"))

# %%
# ----------------------- Processing Dataset Unm 3 -------------------
process_data(unm_3_data, unm_3_labels, os.path.join(
    save_folder, "unm_3.test"), os.path.join(save_folder, "unm_3.labels"), os.path.join(save_folder, "unm_3.indices"))

# %%
# ---------------------- Processing Dataset Cert 1 ------------------
process_data(cert_1_data, cert_1_labels, os.path.join(
    save_folder, "cert_1.test"), os.path.join(save_folder, "cert_1.labels"), os.path.join(save_folder, "cert_1.indices"))

# %%
# --------------------- Processing Dataset Cert 2 --------------------
process_data(cert_2_data, cert_2_labels, os.path.join(
    save_folder, "cert_2.test"), os.path.join(save_folder, "cert_2.labels"), os.path.join(save_folder, "cert_2.indices"))

# %%
# --------------------- Processing Dataset Cert 3 --------------------
process_data(cert_3_data, cert_3_labels, os.path.join(
    save_folder, "cert_3.test"), os.path.join(save_folder, "cert_3.labels"), os.path.join(save_folder, "cert_3.indices"))

# %%


def process_train_set(dataset, data_save_path):
    print(len(dataset))
    min_length = 7
    with open(data_save_path, "w") as data_file:
        for idx, row in enumerate(dataset):
            if(idx % 100 == 0):
                print(f"Processing Entry {idx}")
            count = 0
            letter_count = min_length
            while letter_count <= len(row):
                data_file.write(row[count:count+min_length])
                data_file.write("\n")
                count += 1
                letter_count += 1


# %%
# --------------------- Process Training Set Cert --------------------
train_cert = pd.read_csv(os.path.join(
    cert_dir, "snd-cert.train"), names=["string"])

process_train_set(train_cert["string"],
                  os.path.join(save_folder, "cert.train"))


# %%
# ---------------------- Process Training Set Unm ---------------------
train_unm = pd.read_csv(os.path.join(
    unm_dir, "snd-unm.train"), names=["string"])

process_train_set(train_unm["string"],
                  os.path.join(save_folder, "unm.train"))

# %%
# --------------------- Cert 1 ----------------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_4.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1.csv"), index=False)

# %%
# ---------------------- Cert 2 ----------------------------------
cert2_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_2.test"), names=["chunk"])
cert2_labels = pd.read_csv(os.path.join(
    save_folder, "cert_2.labels"), names=["label"])
cert2_indices = pd.read_csv(os.path.join(
    save_folder, "cert_2.indices"), names=["idx"])
cert2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_2_r_4.txt"), names=["score"])
cert2 = pd.concat([cert2_chunks, cert2_labels,
                   cert2_indices, cert2_scores], axis=1)

cert2_processed = pd.concat([cert_2_data, cert_2_labels], axis=1)
cert2_processed["anomaly_score"] = np.nan
cert2_processed["anomaly_avg"] = np.nan

scores = cert2.groupby(cert2["idx"]).sum()["score"]

for row_idx in range(np.max(cert2["idx"])+1):
    cert2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert2.loc[cert2["idx"] == row_idx, "score"])
cert2_processed.to_csv(os.path.join(
    save_folder, "processed", "cert2.csv"), index=False)

# %%
# -------------------------- Cert 3 -------------------------------
cert3_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_3.test"), names=["chunk"])
cert3_labels = pd.read_csv(os.path.join(
    save_folder, "cert_3.labels"), names=["label"])
cert3_indices = pd.read_csv(os.path.join(
    save_folder, "cert_3.indices"), names=["idx"])
cert3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_3_r_4.txt"), names=["score"])
cert3 = pd.concat([cert3_chunks, cert3_labels,
                   cert3_indices, cert3_scores], axis=1)

cert3_processed = pd.concat([cert_3_data, cert_3_labels], axis=1)
cert3_processed["anomaly_score"] = np.nan
cert3_processed["anomaly_avg"] = np.nan

scores = cert3.groupby(cert3["idx"]).sum()["score"]

for row_idx in range(np.max(cert3["idx"])+1):
    cert3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert3.loc[cert3["idx"] == row_idx, "score"])
cert3_processed.to_csv(os.path.join(
    save_folder, "processed", "cert3.csv"), index=False)

# %%
# ---------------------- Unm 1 ------------------------------------
unm1_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_1.test"), names=["chunk"])
unm1_labels = pd.read_csv(os.path.join(
    save_folder, "unm_1.labels"), names=["label"])
unm1_indices = pd.read_csv(os.path.join(
    save_folder, "unm_1.indices"), names=["idx"])
unm1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_1_r_4.txt"), names=["score"])
unm1 = pd.concat([unm1_chunks, unm1_labels,
                  unm1_indices, unm1_scores], axis=1)

unm1_processed = pd.concat([unm_1_data, unm_1_labels], axis=1)
unm1_processed["anomaly_score"] = np.nan
unm1_processed["anomaly_avg"] = np.nan

scores = unm1.groupby(unm1["idx"]).sum()["score"]

for row_idx in range(np.max(unm1["idx"])+1):
    unm1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm1.loc[unm1["idx"] == row_idx, "score"])
unm1_processed.to_csv(os.path.join(
    save_folder, "processed", "unm1.csv"), index=False)

# %%
# ---------------------- Unm 1 r=3 ------------------------------------
unm1_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_1.test"), names=["chunk"])
unm1_labels = pd.read_csv(os.path.join(
    save_folder, "unm_1.labels"), names=["label"])
unm1_indices = pd.read_csv(os.path.join(
    save_folder, "unm_1.indices"), names=["idx"])
unm1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_1_r_3.txt"), names=["score"])
unm1 = pd.concat([unm1_chunks, unm1_labels,
                  unm1_indices, unm1_scores], axis=1)

unm1_processed = pd.concat([unm_1_data, unm_1_labels], axis=1)
unm1_processed["anomaly_score"] = np.nan
unm1_processed["anomaly_avg"] = np.nan

scores = unm1.groupby(unm1["idx"]).sum()["score"]

for row_idx in range(np.max(unm1["idx"])+1):
    unm1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm1.loc[unm1["idx"] == row_idx, "score"])
unm1_processed.to_csv(os.path.join(
    save_folder, "processed", "unm1r3.csv"), index=False)

# %%
# ---------------------- Unm 1 r=5 ------------------------------------
unm1_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_1.test"), names=["chunk"])
unm1_labels = pd.read_csv(os.path.join(
    save_folder, "unm_1.labels"), names=["label"])
unm1_indices = pd.read_csv(os.path.join(
    save_folder, "unm_1.indices"), names=["idx"])
unm1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_1_r_5.txt"), names=["score"])
unm1 = pd.concat([unm1_chunks, unm1_labels,
                  unm1_indices, unm1_scores], axis=1)

unm1_processed = pd.concat([unm_1_data, unm_1_labels], axis=1)
unm1_processed["anomaly_score"] = np.nan
unm1_processed["anomaly_avg"] = np.nan

scores = unm1.groupby(unm1["idx"]).sum()["score"]

for row_idx in range(np.max(unm1["idx"])+1):
    unm1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm1.loc[unm1["idx"] == row_idx, "score"])
unm1_processed.to_csv(os.path.join(
    save_folder, "processed", "unm1r5.csv"), index=False)

# %%
# ----------------------- Unm 2 -------------------------------
unm2_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_2.test"), names=["chunk"])
unm2_labels = pd.read_csv(os.path.join(
    save_folder, "unm_2.labels"), names=["label"])
unm2_indices = pd.read_csv(os.path.join(
    save_folder, "unm_2.indices"), names=["idx"])
unm2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_2_r_4.txt"), names=["score"])
unm2 = pd.concat([unm2_chunks, unm2_labels,
                  unm2_indices, unm2_scores], axis=1)

unm2_processed = pd.concat([unm_2_data, unm_2_labels], axis=1)
unm2_processed["anomaly_score"] = np.nan
unm2_processed["anomaly_avg"] = np.nan

scores = unm2.groupby(unm2["idx"]).sum()["score"]

for row_idx in range(np.max(unm2["idx"])+1):
    unm2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm2.loc[unm2["idx"] == row_idx, "score"])
unm2_processed.to_csv(os.path.join(
    save_folder, "processed", "unm2.csv"), index=False)

# %%
# ----------------------- Unm 2 r=3 -------------------------------
unm2_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_2.test"), names=["chunk"])
unm2_labels = pd.read_csv(os.path.join(
    save_folder, "unm_2.labels"), names=["label"])
unm2_indices = pd.read_csv(os.path.join(
    save_folder, "unm_2.indices"), names=["idx"])
unm2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_2_r_3.txt"), names=["score"])
unm2 = pd.concat([unm2_chunks, unm2_labels,
                  unm2_indices, unm2_scores], axis=1)

unm2_processed = pd.concat([unm_2_data, unm_2_labels], axis=1)
unm2_processed["anomaly_score"] = np.nan
unm2_processed["anomaly_avg"] = np.nan

scores = unm2.groupby(unm2["idx"]).sum()["score"]

for row_idx in range(np.max(unm2["idx"])+1):
    unm2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm2.loc[unm2["idx"] == row_idx, "score"])
unm2_processed.to_csv(os.path.join(
    save_folder, "processed", "unm2r3.csv"), index=False)

# %%
# ----------------------- Unm 2 r=5 -------------------------------
unm2_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_2.test"), names=["chunk"])
unm2_labels = pd.read_csv(os.path.join(
    save_folder, "unm_2.labels"), names=["label"])
unm2_indices = pd.read_csv(os.path.join(
    save_folder, "unm_2.indices"), names=["idx"])
unm2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_2_r_5.txt"), names=["score"])
unm2 = pd.concat([unm2_chunks, unm2_labels,
                  unm2_indices, unm2_scores], axis=1)

unm2_processed = pd.concat([unm_2_data, unm_2_labels], axis=1)
unm2_processed["anomaly_score"] = np.nan
unm2_processed["anomaly_avg"] = np.nan

scores = unm2.groupby(unm2["idx"]).sum()["score"]

for row_idx in range(np.max(unm2["idx"])+1):
    unm2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm2.loc[unm2["idx"] == row_idx, "score"])
unm2_processed.to_csv(os.path.join(
    save_folder, "processed", "unm2r5.csv"), index=False)

# %%
# -------------------------- Unm 3 ----------------------------
unm3_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_3.test"), names=["chunk"])
unm3_labels = pd.read_csv(os.path.join(
    save_folder, "unm_3.labels"), names=["label"])
unm3_indices = pd.read_csv(os.path.join(
    save_folder, "unm_3.indices"), names=["idx"])
unm3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_3_r_4.txt"), names=["score"])
unm3 = pd.concat([unm3_chunks, unm3_labels,
                  unm3_indices, unm3_scores], axis=1)

unm3_processed = pd.concat([unm_3_data, unm_3_labels], axis=1)
unm3_processed["anomaly_score"] = np.nan
unm3_processed["anomaly_avg"] = np.nan

scores = unm3.groupby(unm3["idx"]).sum()["score"]

for row_idx in range(np.max(unm3["idx"])+1):
    unm3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm3.loc[unm3["idx"] == row_idx, "score"])
unm3_processed.to_csv(os.path.join(
    save_folder, "processed", "unm3.csv"), index=False)

# %%
# -------------------------- Unm 3 r=3 ----------------------------
unm3_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_3.test"), names=["chunk"])
unm3_labels = pd.read_csv(os.path.join(
    save_folder, "unm_3.labels"), names=["label"])
unm3_indices = pd.read_csv(os.path.join(
    save_folder, "unm_3.indices"), names=["idx"])
unm3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_3_r_3.txt"), names=["score"])
unm3 = pd.concat([unm3_chunks, unm3_labels,
                  unm3_indices, unm3_scores], axis=1)

unm3_processed = pd.concat([unm_3_data, unm_3_labels], axis=1)
unm3_processed["anomaly_score"] = np.nan
unm3_processed["anomaly_avg"] = np.nan

scores = unm3.groupby(unm3["idx"]).sum()["score"]

for row_idx in range(np.max(unm3["idx"])+1):
    unm3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm3.loc[unm3["idx"] == row_idx, "score"])
unm3_processed.to_csv(os.path.join(
    save_folder, "processed", "unm3r3.csv"), index=False)

# %%
# -------------------------- Unm 3 r=5 ----------------------------
unm3_chunks = pd.read_csv(os.path.join(
    save_folder, "unm_3.test"), names=["chunk"])
unm3_labels = pd.read_csv(os.path.join(
    save_folder, "unm_3.labels"), names=["label"])
unm3_indices = pd.read_csv(os.path.join(
    save_folder, "unm_3.indices"), names=["idx"])
unm3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_unm_3_r_5.txt"), names=["score"])
unm3 = pd.concat([unm3_chunks, unm3_labels,
                  unm3_indices, unm3_scores], axis=1)

unm3_processed = pd.concat([unm_3_data, unm_3_labels], axis=1)
unm3_processed["anomaly_score"] = np.nan
unm3_processed["anomaly_avg"] = np.nan

scores = unm3.groupby(unm3["idx"]).sum()["score"]

for row_idx in range(np.max(unm3["idx"])+1):
    unm3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    unm3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(unm3.loc[unm3["idx"] == row_idx, "score"])
unm3_processed.to_csv(os.path.join(
    save_folder, "processed", "unm3r5.csv"), index=False)

# %%
# ------------------------- Cert 1 r=3 ----------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_3.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1r3.csv"), index=False)


# %%
# ------------------------ Cert 2 r=3 --------------------------------
cert2_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_2.test"), names=["chunk"])
cert2_labels = pd.read_csv(os.path.join(
    save_folder, "cert_2.labels"), names=["label"])
cert2_indices = pd.read_csv(os.path.join(
    save_folder, "cert_2.indices"), names=["idx"])
cert2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_2_r_3.txt"), names=["score"])
cert2 = pd.concat([cert2_chunks, cert2_labels,
                   cert2_indices, cert2_scores], axis=1)

cert2_processed = pd.concat([cert_2_data, cert_2_labels], axis=1)
cert2_processed["anomaly_score"] = np.nan
cert2_processed["anomaly_avg"] = np.nan

scores = cert2.groupby(cert2["idx"]).sum()["score"]

for row_idx in range(np.max(cert2["idx"])+1):
    cert2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert2.loc[cert2["idx"] == row_idx, "score"])
cert2_processed.to_csv(os.path.join(
    save_folder, "processed", "cert2r3.csv"), index=False)

# %%
# ------------------------ Cert 3 r=3 --------------------------------------
cert3_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_3.test"), names=["chunk"])
cert3_labels = pd.read_csv(os.path.join(
    save_folder, "cert_3.labels"), names=["label"])
cert3_indices = pd.read_csv(os.path.join(
    save_folder, "cert_3.indices"), names=["idx"])
cert3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_3_r_3.txt"), names=["score"])
cert3 = pd.concat([cert3_chunks, cert3_labels,
                   cert3_indices, cert3_scores], axis=1)

cert3_processed = pd.concat([cert_3_data, cert_3_labels], axis=1)
cert3_processed["anomaly_score"] = np.nan
cert3_processed["anomaly_avg"] = np.nan

scores = cert3.groupby(cert3["idx"]).sum()["score"]

for row_idx in range(np.max(cert3["idx"])+1):
    cert3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert3.loc[cert3["idx"] == row_idx, "score"])
cert3_processed.to_csv(os.path.join(
    save_folder, "processed", "cert3r3.csv"), index=False)

# %%
# -------------------------- Cert 1 r=5 -------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_5.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1r5.csv"), index=False)

# %%
# ------------------------ Cert 2 r=5 --------------------------------
cert2_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_2.test"), names=["chunk"])
cert2_labels = pd.read_csv(os.path.join(
    save_folder, "cert_2.labels"), names=["label"])
cert2_indices = pd.read_csv(os.path.join(
    save_folder, "cert_2.indices"), names=["idx"])
cert2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_2_r_5.txt"), names=["score"])
cert2 = pd.concat([cert2_chunks, cert2_labels,
                   cert2_indices, cert2_scores], axis=1)

cert2_processed = pd.concat([cert_2_data, cert_2_labels], axis=1)
cert2_processed["anomaly_score"] = np.nan
cert2_processed["anomaly_avg"] = np.nan

scores = cert2.groupby(cert2["idx"]).sum()["score"]

for row_idx in range(np.max(cert2["idx"])+1):
    cert2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert2.loc[cert2["idx"] == row_idx, "score"])
cert2_processed.to_csv(os.path.join(
    save_folder, "processed", "cert2r5.csv"), index=False)

# %%
# ------------------------ Cert 3 r=5 ------------------------------
cert3_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_3.test"), names=["chunk"])
cert3_labels = pd.read_csv(os.path.join(
    save_folder, "cert_3.labels"), names=["label"])
cert3_indices = pd.read_csv(os.path.join(
    save_folder, "cert_3.indices"), names=["idx"])
cert3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_3_r_5.txt"), names=["score"])
cert3 = pd.concat([cert3_chunks, cert3_labels,
                   cert3_indices, cert3_scores], axis=1)

cert3_processed = pd.concat([cert_3_data, cert_3_labels], axis=1)
cert3_processed["anomaly_score"] = np.nan
cert3_processed["anomaly_avg"] = np.nan

scores = cert3.groupby(cert3["idx"]).sum()["score"]

for row_idx in range(np.max(cert3["idx"])+1):
    cert3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert3.loc[cert3["idx"] == row_idx, "score"])
cert3_processed.to_csv(os.path.join(
    save_folder, "processed", "cert3r5.csv"), index=False)

# %%
# -------------------------- Cert 1 r=6 -------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_6.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1r6.csv"), index=False)

# %%
# ------------------------ Cert 2 r=6 ----------------------------------
cert2_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_2.test"), names=["chunk"])
cert2_labels = pd.read_csv(os.path.join(
    save_folder, "cert_2.labels"), names=["label"])
cert2_indices = pd.read_csv(os.path.join(
    save_folder, "cert_2.indices"), names=["idx"])
cert2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_2_r_6.txt"), names=["score"])
cert2 = pd.concat([cert2_chunks, cert2_labels,
                   cert2_indices, cert2_scores], axis=1)

cert2_processed = pd.concat([cert_2_data, cert_2_labels], axis=1)
cert2_processed["anomaly_score"] = np.nan
cert2_processed["anomaly_avg"] = np.nan

scores = cert2.groupby(cert2["idx"]).sum()["score"]

for row_idx in range(np.max(cert2["idx"])+1):
    cert2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert2.loc[cert2["idx"] == row_idx, "score"])
cert2_processed.to_csv(os.path.join(
    save_folder, "processed", "cert2r6.csv"), index=False)

# %%
# ----------------------- Cert 3 r=6 ---------------------------
cert3_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_3.test"), names=["chunk"])
cert3_labels = pd.read_csv(os.path.join(
    save_folder, "cert_3.labels"), names=["label"])
cert3_indices = pd.read_csv(os.path.join(
    save_folder, "cert_3.indices"), names=["idx"])
cert3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_3_r_5.txt"), names=["score"])
cert3 = pd.concat([cert3_chunks, cert3_labels,
                   cert3_indices, cert3_scores], axis=1)

cert3_processed = pd.concat([cert_3_data, cert_3_labels], axis=1)
cert3_processed["anomaly_score"] = np.nan
cert3_processed["anomaly_avg"] = np.nan

scores = cert3.groupby(cert3["idx"]).sum()["score"]

for row_idx in range(np.max(cert3["idx"])+1):
    cert3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert3.loc[cert3["idx"] == row_idx, "score"])
cert3_processed.to_csv(os.path.join(
    save_folder, "processed", "cert3r5.csv"), index=False)

# %%
# -------------------------- Cert 1 r=7 -------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_7.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1r7.csv"), index=False)

# %%
# ------------------------ Cert 2 r=7 ----------------------------------
cert2_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_2.test"), names=["chunk"])
cert2_labels = pd.read_csv(os.path.join(
    save_folder, "cert_2.labels"), names=["label"])
cert2_indices = pd.read_csv(os.path.join(
    save_folder, "cert_2.indices"), names=["idx"])
cert2_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_2_r_7.txt"), names=["score"])
cert2 = pd.concat([cert2_chunks, cert2_labels,
                   cert2_indices, cert2_scores], axis=1)

cert2_processed = pd.concat([cert_2_data, cert_2_labels], axis=1)
cert2_processed["anomaly_score"] = np.nan
cert2_processed["anomaly_avg"] = np.nan

scores = cert2.groupby(cert2["idx"]).sum()["score"]

for row_idx in range(np.max(cert2["idx"])+1):
    cert2_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert2_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert2.loc[cert2["idx"] == row_idx, "score"])
cert2_processed.to_csv(os.path.join(
    save_folder, "processed", "cert2r7.csv"), index=False)

# %%
# -------------------- Cert 3 r=5 ---------------------------------
cert3_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_3.test"), names=["chunk"])
cert3_labels = pd.read_csv(os.path.join(
    save_folder, "cert_3.labels"), names=["label"])
cert3_indices = pd.read_csv(os.path.join(
    save_folder, "cert_3.indices"), names=["idx"])
cert3_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_3_r_5.txt"), names=["score"])
cert3 = pd.concat([cert3_chunks, cert3_labels,
                   cert3_indices, cert3_scores], axis=1)

cert3_processed = pd.concat([cert_3_data, cert_3_labels], axis=1)
cert3_processed["anomaly_score"] = np.nan
cert3_processed["anomaly_avg"] = np.nan

scores = cert3.groupby(cert3["idx"]).sum()["score"]

for row_idx in range(np.max(cert3["idx"])+1):
    cert3_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert3_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert3.loc[cert3["idx"] == row_idx, "score"])
cert3_processed.to_csv(os.path.join(
    save_folder, "processed", "cert3r5.csv"), index=False)

# %%
# -------------------------- Cert 1 r=2 -------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_2.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1r2.csv"), index=False)

# %%
# -------------------------- Cert 1 r=1 -------------------------
cert1_chunks = pd.read_csv(os.path.join(
    save_folder, "cert_1.test"), names=["chunk"])
cert1_labels = pd.read_csv(os.path.join(
    save_folder, "cert_1.labels"), names=["label"])
cert1_indices = pd.read_csv(os.path.join(
    save_folder, "cert_1.indices"), names=["idx"])
cert1_scores = pd.read_csv(os.path.join(
    save_folder, "final_anomaly_scores", "anomaly_scores_cert_1_r_1.txt"), names=["score"])
cert1 = pd.concat([cert1_chunks, cert1_labels,
                   cert1_indices, cert1_scores], axis=1)

cert1_processed = pd.concat([cert_1_data, cert_1_labels], axis=1)
cert1_processed["anomaly_score"] = np.nan
cert1_processed["anomaly_avg"] = np.nan

scores = cert1.groupby(cert1["idx"]).sum()["score"]

for row_idx in range(np.max(cert1["idx"])+1):
    cert1_processed.loc[row_idx, "anomaly_score"] = scores[row_idx]
    cert1_processed.loc[row_idx, "anomaly_avg"] = scores[row_idx] / \
        len(cert1.loc[cert1["idx"] == row_idx, "score"])
cert1_processed.to_csv(os.path.join(
    save_folder, "processed", "cert1r1.csv"), index=False)

# %%
