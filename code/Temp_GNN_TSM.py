import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 1) load ───────────────────────────────

output_csv_path = "aggregated_sales.csv"
df = pd.read_csv(output_csv_path, parse_dates=["date"]).set_index("date")
# shape: (T, N)  e.g. (800, 50)

# 2) add calendar covariates  (sin/cos, weekday)
cal = pd.DataFrame({
    "sin_doy":  np.sin(2*np.pi*df.index.dayofyear/365),
    "cos_doy":  np.cos(2*np.pi*df.index.dayofyear/365),
    "weekday":  df.index.weekday
}, index=df.index)

full = pd.concat([df, cal], axis=1)

# 3) scale each column separately
scaler = StandardScaler()
X = scaler.fit_transform(full)
X = torch.tensor(X, dtype=torch.float32)          # (T, N + 3)

# 4) sliding-window dataset
HIST, HORIZON = 30, 7          # last 30 days → next 7   (modifiable)
seq, tgt = [], []
for t in range(HIST, len(X) - HORIZON + 1):
    seq.append(X[t-HIST:t])                # (30, N+3)
    tgt.append(X[t:t+HORIZON, :df.shape[1]])  # only sales part as target
seq  = torch.stack(seq)                    # (samples, 30, D)
tgt  = torch.stack(tgt)                    # (samples, 7, N)

class SparseGraphLearner(nn.Module):
    def __init__(self, n_nodes, l1_alpha=1e-3):
        super().__init__()
        self.A_logits = nn.Parameter(torch.randn(n_nodes, n_nodes))
        self.l1 = l1_alpha          # sparsity weight
    def forward(self):
        A = torch.sigmoid(self.A_logits)      # (0,1)
        A = A * (1 - torch.eye(A.size(0)))     # remove self-loops
        return A
    def l1_loss(self):
        return self.l1 * torch.abs(torch.sigmoid(self.A_logits)).sum()

##############
from torch_geometric.nn import GCNConv
class TGNN(nn.Module):
    def __init__(self, n_series, hidden=64, horizon=HORIZON):
        super().__init__()
        self.glearner = SparseGraphLearner(n_series)
        self.gc1 = GCNConv(in_channels=n_series+3, out_channels=hidden)
        self.tcn = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2, padding=2)
        self.head = nn.Linear(hidden, horizon)   # step-ahead per node
    def forward(self, seq):       # seq: (B, L, D)  where D=N+3
        A = self.glearner()       # (N,N)
        edge_index = A.nonzero().t()        # COO indices
        edge_weight = A[edge_index[0], edge_index[1]]

        # reshape for graph conv: treat every time step separately
        B, L, D = seq.shape
        x = seq.reshape(B*L, D)                  # nodes=batch*L
        x = self.gc1(x, edge_index, edge_weight)
        x = torch.relu(x).reshape(B, L, -1).permute(0,2,1)  # (B, hidden, L)

        h = torch.relu(self.tcn(x))              # temporal features
        h = torch.mean(h, dim=-1)                # (B, hidden)
        out = self.head(h)                       # (B, horizon)
        return out, A

###########################################################
#  Training loop (device-agnostic: CPU or GPU works alike) #
###########################################################
# ──────────────────────────────────────────────────────────
# Split seq / tgt time series samples into train / val / test
#     - chronological split to avoid data leakage
# ──────────────────────────────────────────────────────────
TOTAL = len(seq)
train_end = int(0.7 * TOTAL)            # 70% for training
val_end   = int(0.85 * TOTAL)           # 15% validation, 15% test

seq_train, tgt_train = seq[:train_end], tgt[:train_end]
seq_val,   tgt_val   = seq[train_end:val_end], tgt[train_end:val_end]
seq_test,  tgt_test  = seq[val_end:], tgt[val_end:]

print(f"train:{len(seq_train)}, val:{len(seq_val)}, test:{len(seq_test)}")

# ──────────────────────────────────────────────────────────
# Device setup + training loop
# ──────────────────────────────────────────────────────────
import torch, torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> training on", DEVICE)

model   = TGNN(n_series=df.shape[1]).to(DEVICE)
opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.L1Loss()                      # MAE in scaled space
BATCH   = 64

def batch_iter(x, y, bs):
    idx = torch.randperm(len(x))
    for i in range(0, len(x), bs):
        j   = idx[i:i+bs]
        yield x[j].to(DEVICE), y[j].to(DEVICE)

for epoch in range(80):
    # ---- training ----
    model.train(); train_mae = 0.0
    for xb, yb in batch_iter(seq_train, tgt_train, BATCH):
        opt.zero_grad()
        pred, _   = model(xb)
        loss      = loss_fn(pred, yb.mean(-1)) + model.glearner.l1_loss()
        loss.backward(); opt.step()
        train_mae += loss_fn(pred, yb.mean(-1)).item() * len(xb)

    # ---- validation ----
    model.eval(); val_mae = 0.0
    with torch.no_grad():
        for xb, yb in batch_iter(seq_val, tgt_val, BATCH):
            pred, _ = model(xb)
            val_mae += loss_fn(pred, yb.mean(-1)).item() * len(xb)

    train_mae /= len(seq_train)
    val_mae   /= len(seq_val)
    print(f"E{epoch:02d}  train-MAE {train_mae:.4f} | val-MAE {val_mae:.4f}")

#########
# ───────────────────────────────
# 1) Compute overall average scaling factor (a single scalar)
# ───────────────────────────────
avg_scale = torch.tensor(scaler.scale_[:df.shape[1]].mean(), dtype=torch.float32)
avg_mean  = torch.tensor(scaler.mean_ [:df.shape[1]].mean(),  dtype=torch.float32)

# ───────────────────────────────
# 2) Prediction & inverse transform
# ───────────────────────────────
model.eval(); preds, trues = [], []
with torch.no_grad():
    for xb, yb in batch_iter(seq_test, tgt_test, BATCH):
        p, _ = model(xb.to(DEVICE))
        preds.append(p.cpu())
        trues.append(yb.mean(-1).cpu())      # consistent with training

preds = torch.cat(preds) * avg_scale + avg_mean     # (samples, horizon)
trues = torch.cat(trues) * avg_scale + avg_mean

# ───────────────────────────────
# 3) Compute out-of-time MAPE
# ───────────────────────────────
mape = ((preds - trues).abs() / trues.clamp(min=1e-8)).mean().item() * 100
print(f"Out-of-time MAPE = {mape:.2f} %")

A = model.glearner().cpu().detach().numpy()
import networkx as nx, matplotlib.pyplot as plt
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
# threshold for visibility
G = nx.DiGraph( (u,v,d) for u,v,d in G.edges(data=True) if d['weight']>0.15 )
nx.draw(G, node_size=300, arrows=True)

# =========================================================
# 5.  Classical baselines: SARIMAX & SES for comparison
# =========================================================
# -----------------------------------------------------------
# generic one-step-ahead rolling forecaster  (works for both)
# -----------------------------------------------------------
def rolling_forecast(model_builder, history, steps, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    hist  = list(history)
    out   = []
    for _ in range(steps):
        model   = model_builder(hist)          # build fresh model
        result  = model.fit(**fit_kwargs)
        fc      = result.forecast(1)[0]
        out.append(fc)
        hist.append(fc)                        # roll the window
    return np.array(out)

test_len  = len(seq_te)                        # number of rolling steps
train_raw = df.iloc[:val_end + HIST]           # data up to test start

sarimax_preds, ses_preds = [], []
for col in df.columns:
    trn = train_raw[col].values
    # SARIMAX(1,0,1)
    sarimax_preds.append(
        rolling_forecast(
            lambda h: SARIMAX(h, order=(1,0,1)),
            trn, test_len,
            fit_kwargs={"disp": False}
        )
    )
    # Simple Exponential Smoothing
    ses_preds.append(
        rolling_forecast(
            lambda h: SimpleExpSmoothing(h, initialization_method="estimated"),
            trn, test_len
        )
    )

sarimax_preds = np.column_stack(sarimax_preds).mean(axis=1)   # align shapes
ses_preds     = np.column_stack(ses_preds).mean(axis=1)

# ---------- MAPE helper ----------
def mape(pred, true):
    return np.mean(np.abs(pred-true) / np.clip(true, 1e-8, None)) * 100

truth_avg   = truth.mean(axis=1)           # (samples,)

sx_mape  = mape(sarimax_preds, truth_avg)
ses_mape = mape(ses_preds,     truth_avg)

print("\n=============  OUT-OF-TIME  MAPE  =============")
print(f"SARIMAX(1,0,1)  : {sx_mape:6.2f} %")
print(f"SES             : {ses_mape:6.2f} %")