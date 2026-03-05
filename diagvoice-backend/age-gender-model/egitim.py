import os, json, torch, torchaudio, numpy as np, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

MODEL_DIR  = os.path.join(OUTPUT_DIR, "final_model_fixed")
TEST_CSV   = os.path.join(OUTPUT_DIR, "test_manifest.csv")

SR=16000
MAX_LEN=SR*4
BASE_MODEL="microsoft/wavlm-base"

# load label maps
with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
    maps = json.load(f)
gender_labels = maps["gender"]
agebin_labels = maps["agebin"]

gender2id = {k:i for i,k in enumerate(gender_labels)}
agebin2id = {k:i for i,k in enumerate(agebin_labels)}

test_df = pd.read_csv(TEST_CSV)
test_df["gender"] = test_df["gender"].astype(str).str.lower().str.strip()
test_df["age_bin"] = test_df["age_bin"].astype(str).str.strip()
test_df["gender_id"] = test_df["gender"].map(gender2id)
test_df["agebin_id"] = test_df["age_bin"].map(agebin2id)
test_df = test_df.dropna(subset=["gender_id","agebin_id","path"])
test_df["gender_id"] = test_df["gender_id"].astype(int)
test_df["agebin_id"] = test_df["agebin_id"].astype(int)

class AudioDS(Dataset):
    def __init__(self, df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        wav,sr=torchaudio.load(r["path"])
        wav=wav.mean(dim=0)
        if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
        wav=torch.nan_to_num(wav)
        wav = wav[:MAX_LEN] if wav.numel()>MAX_LEN else nn.functional.pad(wav,(0,MAX_LEN-wav.numel()))
        return {"input_values": wav, "g": int(r["gender_id"]), "a": int(r["agebin_id"])}

def collate(b):
    return {
        "input_values": torch.stack([x["input_values"] for x in b]),
        "g": torch.tensor([x["g"] for x in b]),
        "a": torch.tensor([x["a"] for x in b]),
    }

# model definition (same heads)
class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(BASE_MODEL)
        h=self.encoder.config.hidden_size
        self.drop=nn.Dropout(0.1)
        self.g_head=nn.Linear(h,len(gender_labels))
        self.a_head=nn.Linear(h,len(agebin_labels))
    def forward(self,input_values):
        x=self.encoder(input_values=input_values).last_hidden_state.mean(dim=1)
        x=self.drop(x)
        return self.g_head(x), self.a_head(x)

model=MultiTask()
state=torch.load(os.path.join(MODEL_DIR,"model.pt"), map_location="cpu")
model.load_state_dict(state, strict=True)

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev); model.eval()

ds=AudioDS(test_df)
dl=DataLoader(ds,batch_size=8,shuffle=False,collate_fn=collate)

gT,gP,aT,aP=[],[],[],[]
with torch.no_grad():
    for b in dl:
        iv=b["input_values"].to(dev)
        lg,la=model(iv)
        gT += b["g"].tolist()
        aT += b["a"].tolist()
        gP += lg.argmax(-1).cpu().tolist()
        aP += la.argmax(-1).cpu().tolist()

gT=np.array(gT); gP=np.array(gP); aT=np.array(aT); aP=np.array(aP)

def macro_f1(y,p,n):
    f=[]
    for c in range(n):
        tp=np.sum((y==c)&(p==c))
        fp=np.sum((y!=c)&(p==c))
        fn=np.sum((y==c)&(p!=c))
        pr=tp/(tp+fp) if tp+fp else 0
        rc=tp/(tp+fn) if tp+fn else 0
        f.append((2*pr*rc)/(pr+rc) if pr+rc else 0)
    return float(np.mean(f))

def cm(y,p,n):
    m=np.zeros((n,n),dtype=int)
    for yt,yp in zip(y,p): m[int(yt),int(yp)]+=1
    return m

def plot_cm(M, labels, title, path):
    plt.figure(figsize=(6,5))
    plt.imshow(M); plt.title(title); plt.colorbar()
    plt.xticks(range(len(labels)),labels,rotation=45,ha="right")
    plt.yticks(range(len(labels)),labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j,i,str(M[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(path,dpi=200); plt.close()

metrics = {
    "gender_acc": float((gT==gP).mean()),
    "agebin_acc": float((aT==aP).mean()),
    "gender_macro_f1": macro_f1(gT,gP,len(gender_labels)),
    "agebin_macro_f1": macro_f1(aT,aP,len(agebin_labels)),
}

cm_g = cm(gT,gP,len(gender_labels))
cm_a = cm(aT,aP,len(agebin_labels))

plot_cm(cm_g, gender_labels, "Gender CM (Test)", os.path.join(OUTPUT_DIR,"cm_gender_test.png"))
plot_cm(cm_a, agebin_labels, "AgeBin CM (Test)", os.path.join(OUTPUT_DIR,"cm_agebin_test.png"))

with open(os.path.join(OUTPUT_DIR,"test_metrics.json"),"w",encoding="utf-8") as f:
    json.dump(metrics,f,ensure_ascii=False,indent=2)

print("✅ Test metrics:", metrics)
print("✅ Saved:", os.path.join(OUTPUT_DIR,"test_metrics.json"))