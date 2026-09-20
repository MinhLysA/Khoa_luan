# CLAUDE CODE IMPLEMENTATION BRIEF — KLTN RL INVENTORY

## 0. Mục đích tài liệu

Tài liệu này là **source-of-truth** để Claude Code chỉnh sửa project `rl_inventory` trước khi chạy lại training chính thức và viết báo cáo KLTN bằng LaTeX.

**Đề tài tiếng Việt:**  
**Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho**

**Tên tiếng Anh dùng trong báo cáo:**  
**Developing a Reinforcement Learning Model for Optimizing Reorder Policies in Multi-Warehouse Inventory Management**

Có thể dùng thuật ngữ `replenishment policy` trong phần học thuật khi mô tả quyết định bổ sung tồn kho, nhưng **không tự ý đổi tên đề tài đã đăng ký**.

---

# 1. Phạm vi nghiên cứu phải giữ nguyên

Project hiện tại phải được định hình là:

> **M5-based multi-warehouse, multi-SKU inventory simulation + stochastic lead time + warehouse capacity + Parameter-Sharing IPPO + cost/service-level optimization + traditional baselines + robustness/statistical evaluation.**

### Phạm vi bắt buộc giữ

- Dữ liệu nhu cầu: **M5 Forecasting**.
- 10 demand nodes được ánh xạ thành **10 virtual warehouses**.
- 30 SKU đại diện.
- Tổng số decision agents:

\[
N = 10 \times 30 = 300.
\]

- Mỗi agent tương ứng với một cặp:

\[
i=(w,k)
\]

trong đó `w` là warehouse và `k` là SKU.

- Thuật toán chính: **Parameter-Sharing Independent Proximal Policy Optimization (PS-IPPO)**.
- Mỗi agent đưa ra quyết định replenishment độc lập từ local observation.
- Các agent dùng chung Actor và Critic parameters.
- Không có coordinator agent.
- Không có inter-warehouse transfer.
- Không có central warehouse.
- Không phải multi-echelon.

### Không tự ý mở rộng thành

- MAPPO/HAPPO/QMIX/VDN làm thuật toán chính.
- Multi-echelon supply chain.
- Central warehouse → regional warehouse.
- Lateral transshipment.
- Deep demand forecasting/GNN/Transformer làm core contribution.
- Slotting optimization.
- Product embedding/new-product generalization.
- MCP làm thành phần cốt lõi của RL.

Các nội dung trên chỉ có thể đưa vào **Related Work / Future Work** trừ khi được yêu cầu riêng sau này.

---

# 2. Kiến trúc Multi-Agent phải hiểu đúng

## 2.1 Agent

Một agent là một **warehouse–SKU decision entity**.

Ví dụ:

```text
Agent(CA_1, FOODS_001)
Agent(CA_1, FOODS_002)
...
Agent(WI_3, SKU_030)
```

Tổng:

```text
10 warehouses × 30 SKUs = 300 agents
```

## 2.2 Parameter sharing

Không được mô tả là “300 neural networks riêng”.

Cách diễn đạt đúng:

- Có 300 tác tử ra quyết định.
- Các tác tử là homogeneous agents.
- Tất cả sử dụng chung một Actor:

\[
\pi_\theta(a_i|o_i)
\]

- Tất cả sử dụng chung một Critic cục bộ:

\[
V_\phi(o_i)
\]

- Independence nằm ở **decision process**: mỗi agent quan sát local observation và sinh action riêng.
- Parameter sharing chỉ chia sẻ trọng số `theta`, `phi`, không hợp nhất 300 decision entities thành một agent.

## 2.3 Coordinator

**Không thêm Coordinator Agent.**

Environment và training loop không phải decision-making agents.

Cách giải thích:

> Mỗi warehouse–SKU agent thực thi phi tập trung. Các agent trong cùng warehouse tương tác gián tiếp qua shared warehouse capacity. Environment quản lý inventory dynamics, pipeline, demand, capacity và cost. Training module tổng hợp trajectories để cập nhật shared policy.

---

# 3. Business Logic chuẩn để giữ đồng nhất code và báo cáo

```text
M5 demand + calendar + price
            ↓
Khởi tạo inventory state
            ↓
Ngày t
            ↓
Tạo observation cho 300 warehouse–SKU agents
            ↓
Shared PS-IPPO Actor sinh 300 actions
            ↓
Action → replenishment quantity
            ↓
Hàng từ pipeline đến sau stochastic lead time
            ↓
Capacity acceptance / overflow rejection
            ↓
Demand ngày t phát sinh
            ↓
Sales / stockout
            ↓
Update inventory + fill rate + demand history
            ↓
Holding + stockout + ordering + overflow cost
            ↓
Training reward
            ↓
State ngày t+1
```

---

# 4. Mục tiêu kinh doanh và Reward phải tách riêng

## 4.1 Business objective

Mục tiêu nghiên cứu cuối cùng:

\[
\min_{\pi} J(\pi)
\]

với:

\[
J = C_H + C_S + C_O + C_V
\]

subject to:

\[
FillRate \ge \beta
\]

với:

\[
\beta = 0.85.
\]

Trong đó:

- `C_H`: holding cost.
- `C_S`: stockout/lost-sales cost.
- `C_O`: ordering cost.
- `C_V`: overflow/capacity violation cost.

## 4.2 Training reward

Reward cho agent có thể chứa thêm service-level shaping:

\[
r_{i,t} = -\left(
C^H_{i,t}+C^S_{i,t}+C^O_{i,t}+C^V_{i,t}+P^{SL}_{i,t}
\right).
\]

Trong đó:

\[
P^{SL}_{i,t}
= \phi\,p\,\max(0,\beta-FR_{i,t})\,\bar d_i.
\]

**Không cộng service penalty vào economic cost cuối cùng khi báo cáo.**

Economic cost để so sánh:

\[
C^{economic}=C_H+C_S+C_O+C_V.
\]

Reward âm không có nghĩa model xấu. Nếu reward là `-cost`, giá trị ít âm hơn nghĩa là chi phí thấp hơn.

---

# 5. P0 — bắt buộc sửa trước khi train lại

---

## P0.1 Sửa lỗi overflow/capacity

### File

```text
env/inventory_env.py
```

### Lỗi hiện tại

Ở B2 code đang:

```python
hang_ve = self.pipeline_orders[0].copy()
self.inventory = self.inventory + hang_ve

inv_wh   = self.inventory.reshape(self.n_warehouses, self.n_skus)
tong_kho = inv_wh.sum(axis=1, keepdims=True)
cap      = self.suc_chua_kho[:, None]
vuot     = np.maximum(0.0, tong_kho - cap)
ty_le    = inv_wh / np.maximum(tong_kho, 1e-6)
overflow = (vuot * ty_le).reshape(-1).astype(np.float32)
self.inventory = (inv_wh - vuot * ty_le).reshape(-1).astype(np.float32)
```

Logic này phân bổ overflow trên **inventory cũ + incoming**, nên có thể vô tình loại cả tồn kho cũ.

### Logic đúng

Chỉ được reject phần hàng **vừa đến**.

Nếu:

```text
capacity = 100
old inventory = 90
incoming = 20
```

thì:

```text
accepted incoming = 10
rejected incoming = 10
final inventory = 100
```

Không được xóa tồn kho cũ.

### Patch đề xuất

Thay B2 bằng logic tương đương:

```python
# -- B2: Nhan hang tu pipeline --------------------------------------
hang_ve = self.pipeline_orders[0].copy()

inv_before_wh = self.inventory.reshape(
    self.n_warehouses, self.n_skus
)
incoming_wh = hang_ve.reshape(
    self.n_warehouses, self.n_skus
)

used_capacity = inv_before_wh.sum(axis=1)
free_capacity = np.maximum(
    self.suc_chua_kho - used_capacity, 0.0
)

incoming_total = incoming_wh.sum(axis=1)

accept_ratio = np.ones(
    self.n_warehouses, dtype=np.float32
)

mask = incoming_total > free_capacity
accept_ratio[mask] = (
    free_capacity[mask]
    / np.maximum(incoming_total[mask], 1e-6)
)

accepted_wh = incoming_wh * accept_ratio[:, None]
rejected_wh = incoming_wh - accepted_wh

self.inventory = (
    inv_before_wh + accepted_wh
).reshape(-1).astype(np.float32)

overflow = rejected_wh.reshape(-1).astype(np.float32)
```

### Công thức báo cáo

\[
FreeCapacity_{w,t}
=
\max\left(0,Cap_w-\sum_i I_{w,i,t}\right)
\]

\[
\rho_{w,t}
=
\min\left(
1,
\frac{FreeCapacity_{w,t}}
{\sum_i Incoming_{w,i,t}+\epsilon}
\right)
\]

\[
Accepted_{w,i,t}=\rho_{w,t}Incoming_{w,i,t}
\]

\[
Overflow_{w,i,t}=Incoming_{w,i,t}-Accepted_{w,i,t}
\]

### Info output nên tách

Thêm nếu phù hợp:

```python
"received_attempted": float(hang_ve.sum()),
"received_accepted": float(accepted_wh.sum()),
"overflow": float(overflow.sum()),
```

Không phá các key cũ nếu dashboard đang dùng; có thể giữ `received` để backward compatible.

---

## P0.2 Thêm regression tests cho capacity

### File

```text
tests/test_env.py
```

Thêm test tối thiểu cho case:

```text
old inventory = 90
capacity = 100
incoming = 20
```

Acceptance criteria:

- inventory cũ không bị giảm bởi capacity rejection.
- accepted incoming = 10.
- overflow = 10.
- final inventory trước demand = 100.
- inventory không âm.
- tổng flow được bảo toàn:

\[
OldInventory + Incoming
=
FinalPreDemandInventory + Overflow.
\]

Nếu test cần disable demand để kiểm tra chính xác, thiết kế test fixture hoặc monkeypatch demand về 0.

---

## P0.3 Chốt cost model cho main experiment

### File

```text
config.yaml
```

Main experiment nên dùng normalized simulation cost để tránh khẳng định các cost này là chi phí thực của Walmart.

Đề xuất:

```yaml
env:
  cp_lk: 1.0
  cp_th: 10.0
  cp_dh: 5.0
  pt_tk: 5.0

  phi_dv: 3.0
  muc_dv: 0.85

  use_real_price_stockout: false
```

Giữ `price_series` trong observation để phản ánh discount/promotion signal nếu code đang dùng.

### Không ghi đơn vị USD trong báo cáo

Gọi là:

> normalized cost units / hệ số chi phí chuẩn hóa.

Lý do:

M5 không cung cấp trực tiếp:

- holding cost thực tế,
- supplier ordering cost,
- purchase cost,
- true stockout/lost-margin cost,
- warehouse capacity,
- lead time.

Những biến này là simulation assumptions.

### Sensitivity sau main experiment

Không cần trước khi train main model.

Sau đó có thể chạy:

\[
\frac{p}{h}\in\{5,10,20\}
\]

hoặc các kịch bản cost khác để kiểm tra robustness.

---

## P0.4 Sửa true episode fill rate trong training validation

### File

```text
scripts/train.py
```

### Vấn đề

Hiện code tính:

```python
ep_f.append(inf["fill_rate_mean"])
...
fills.append(float(np.mean(ep_f)))
```

Đây là trung bình của rolling fill-rate signal, không phải fill rate của toàn evaluation horizon.

### Công thức đúng

\[
FillRate
=
1-
\frac{\sum_t Stockout_t}
{\sum_t Demand_t}.
\]

### Sửa `run_deterministic_eval()`

Hàm cần trả về:

```text
mean reward
mean economic cost
mean true episode fill rate
```

Logic đề xuất:

```python
def run_deterministic_eval():
    rewards = []
    costs = []
    fills = []

    for i in range(eval_n_episodes):
        o, _ = eval_env.reset(seed=90000 + i)

        ep_reward = 0.0
        ep_cost = 0.0
        demand_total = 0.0
        stockout_total = 0.0

        while True:
            a, _, _ = agent.select_action(
                o, deterministic=True
            )

            o, _, term, trunc, inf = eval_env.step(a)

            ep_reward += inf["raw_reward"]

            ep_cost += (
                inf["cost_holding"]
                + inf["cost_stockout"]
                + inf["cost_ordering"]
                + inf["cost_overflow"]
            )

            demand_total += inf["demand"]
            stockout_total += inf["stockout"]

            if term or trunc:
                break

        fill = 1.0 - (
            stockout_total / max(demand_total, 1e-6)
        )

        rewards.append(ep_reward)
        costs.append(ep_cost)
        fills.append(fill)

    return (
        float(np.mean(rewards)),
        float(np.mean(costs)),
        float(np.mean(fills)),
    )
```

Tên biến và return signature có thể điều chỉnh nhưng behavior phải đúng.

---

## P0.5 Chọn best checkpoint theo constrained business objective

### File

```text
scripts/train.py
config.yaml
```

### Vấn đề

Hiện:

```python
if det_f >= min_fill_to_save and det_r > best_score:
    save()
```

và:

```yaml
min_fill_to_save: 0.0
```

Điều này không khớp với thesis objective `FillRate >= 85%`.

### Sửa config

```yaml
ppo:
  min_fill_to_save: 0.85
```

### Cách tốt hơn

Chọn checkpoint có **economic cost thấp nhất trong nhóm đạt service constraint**.

Khởi tạo:

```python
best_feasible_cost = float("inf")
```

Sau deterministic validation:

```python
det_reward, det_cost, det_fill = run_deterministic_eval()

if det_fill >= min_fill_to_save:
    if det_cost < best_feasible_cost:
        best_feasible_cost = det_cost
        agent.save(...)
```

Nếu chưa có checkpoint nào feasible, có thể giữ fallback tốt nhất theo fill-rate/reward để tránh không có model file, nhưng phải log rõ `feasible=False`.

### Logging

Nên log:

```text
eval_reward
eval_cost
eval_fill
checkpoint_feasible
```

---

## P0.6 Training log phải dùng true episode fill rate

### File

```text
scripts/train.py
```

Hiện training episode dùng:

```python
ep_fill.append(info["fill_rate_mean"])
```

Thay cho KPI business bằng:

```python
ep_demand = 0.0
ep_stockout = 0.0
```

mỗi step:

```python
ep_demand += info["demand"]
ep_stockout += info["stockout"]
```

cuối episode:

```python
episode_fill = 1.0 - ep_stockout / max(ep_demand, 1e-6)
```

Dùng `episode_fill` cho:

- `train_log.csv` column `fill_rate`.
- TensorBoard `train/fill_rate`.
- progress print.

Có thể giữ rolling fill-rate signal nội bộ trong state/reward, nhưng không dùng nó làm KPI episode.

---

# 6. P1 — phải sửa trước final evaluation

---

## P1.1 Tune baseline trên validation, không tune trên test

### File

```text
scripts/tune_baselines.py
```

### Hiện tại

Comment và code đang tune trên `mode="train"`.

### Yêu cầu mới

Baseline hyperparameters dùng **validation split**, tương tự checkpoint selection của RL.

```text
Train      → học RL policy parameters
Validation → chọn RL checkpoint + tune baseline hyperparameters
Test       → chỉ báo cáo kết quả cuối cùng
```

Đổi environment tuning sang:

```python
env = MultiWarehouseInventoryEnv(
    ...,
    mode="val"
)
```

### Đồng bộ input environment

`tune_baselines.py` phải load cùng các data feature với RL environment nếu feature/cost phụ thuộc chúng:

```text
demand_data.npy
calendar_features.npy
price_per_pair.npy
price_series.npy
```

Ngay cả khi main experiment dùng `use_real_price_stockout=false`, nên đồng bộ loader để tránh divergence về sau.

### Selection rule

Hai lựa chọn:

1. `baseline_params.json`: tìm config có economic cost thấp nhất trên validation không ràng buộc service.
2. `iso_service.py`: tìm config có economic cost thấp nhất với `fill >= target`.

Giữ hai mục đích này tách rõ.

---

## P1.2 Sửa `iso_service.py`

### File

```text
scripts/iso_service.py
```

### Vấn đề 1 — target thấp hơn SLA

Hiện:

```python
target = ippo_fill - args.tolerance
```

Sửa:

```python
target = max(
    env_te.muc_dv,
    ippo_fill - args.tolerance
)
```

Như vậy target không bao giờ thấp hơn 85%.

### Vấn đề 2 — tuning split

Hiện script ghi “tìm kiếm trên miền TRAIN”.

Đổi thành validation environment để không tune hyperparameters trên train hoặc test.

Tên biến nên đổi rõ:

```text
env_val
env_test
```

### Bảng final iso-service

Báo cáo:

```text
Policy | Economic Cost | Fill Rate | Feasible? | Delta vs IPPO
```

---

## P1.3 Statistical testing phải là paired

### File

```text
scripts/evaluate.py
```

### Vấn đề

Hiện dùng:

```python
stats.ttest_ind(..., equal_var=False)
```

nhưng các policy được chạy trên cùng seed/scenario, nên observations được paired.

### Sửa

```python
n = min(len(ppo_cost), len(base_cost))
ppo_cost = ppo_cost[:n]
base_cost = base_cost[:n]

diff = ppo_cost - base_cost

t_stat, p_t = stats.ttest_rel(
    ppo_cost,
    base_cost
)

try:
    _, p_w = stats.wilcoxon(diff)
except ValueError:
    p_w = float("nan")

cohen_dz = (
    diff.mean()
    / (diff.std(ddof=1) + 1e-9)
)
```

### JSON keys

Nên đổi:

```text
cohen_d → cohen_dz
p_ttest → p_paired_ttest
```

Nếu cần backward compatibility, giữ thêm key cũ nhưng ghi rõ method mới.

### Công thức báo cáo

\[
\Delta C_j
=
C^{IPPO}_j-C^{Baseline}_j.
\]

Paired t-test:

\[
H_0:\mu_{\Delta C}=0.
\]

Wilcoxon signed-rank dùng như non-parametric check.

---

## P1.4 Final validation/test horizon không nên random overlapping 365-day windows

### File

```text
env/inventory_env.py
```

### Hiện tại

- `episode_length = 365`.
- test split có 491 ngày.
- mỗi evaluation episode random `start_day`.
- 30 episodes chồng lấn demand rất mạnh.

### Yêu cầu

Training vẫn có thể dùng random windows 365 ngày.

Validation/Test nên hỗ trợ **fixed full split horizon**:

```text
Train:      [0, 1050)
Validation: [1050, 1450) = 400 days
Test:       [1450, 1941) = 491 days
```

Nên thêm runtime property:

```python
self.current_episode_length = self.episode_length
```

Trong `reset()`:

```python
if self.mode == "test":
    self.start_day = self.val_day
    self.current_episode_length = T - self.val_day
elif self.mode == "val":
    self.start_day = self.split_day
    self.current_episode_length = self.val_day - self.split_day
else:
    # train random 365-day window
    ...
```

Trong `step()`:

```python
truncated = (
    self.current_step
    >= self.current_episode_length
)
```

### Replications

Nếu chạy 30 replications trên fixed test demand horizon, chỉ thay stochastic sources như random lead time.

Mỗi policy phải dùng **cùng replication seed**.

---

## P1.5 Warm-start demand history khi reset

### File

```text
env/inventory_env.py
```

### Vấn đề

Hiện:

```python
self.demand_history = np.zeros(...)
```

nên đầu episode agent nhìn thấy 7 ngày demand = 0 dù dữ liệu lịch sử tồn tại.

### Sửa

Sau khi chọn `start_day`, nạp history trước `start_day`:

```python
hist_start = max(
    0,
    self.start_day - self.lookback
)

hist = self.demand_data[
    hist_start:self.start_day
].reshape(-1, self.n_pairs)

self.demand_history[:] = 0.0

if len(hist) > 0:
    self.demand_history[-len(hist):] = hist
```

Đây không phải data leakage vì chỉ sử dụng dữ liệu quá khứ tại thời điểm ra quyết định.

### Có thể cân nhắc warm-start fill-rate window

Không bắt buộc P1. Nếu làm, phải chỉ dùng historical demand/simulated historical fulfillment hợp lệ. Không tự sinh stockout history từ future data.

---

## P1.6 Multiseed phải dùng cùng training budget

### File

```text
run.py
```

Hiện:

```python
ms_episodes = a.episodes or 1000
```

Sửa:

```python
ms_episodes = (
    a.episodes
    or cfg["ppo"]["total_episodes"]
)
```

Final multiseed tối thiểu:

```text
seed 42
seed 1
seed 2
```

mỗi seed:

```text
5000 episodes
```

Nếu compute cho phép có thể mở rộng 5 seeds sau.

Không được so sánh “main 5000 episodes” với “multiseed 1000 episodes” rồi gọi đó là reliability của model final.

---

# 7. P2 — thực nghiệm bổ sung sau khi main model ổn

Không triển khai trước khi P0/P1 pass tests và main PS-IPPO đã train ổn.

---

## P2.1 Demand regime experiment

### Mục tiêu

Kiểm tra góp ý GVHD:

- Classical policy có thể mạnh trong demand ổn định.
- RL có thể thích nghi tốt hơn khi demand biến động.

### Script mới đề xuất

```text
scripts/evaluate_demand_regimes.py
```

### Cách chia regime

Dùng rolling window, ví dụ 28 ngày.

Tính:

\[
CV_j
=
\frac{\sigma(D_j)}{\mu(D_j)+\epsilon}.
\]

Phân nhóm:

```text
Stable     = low CV
Volatile   = high CV
Event      = windows có M5 event/SNAP signal
```

Có thể dùng quantile để chia low/high CV, ví dụ Q25 và Q75.

Không hard-code kết luận RL thắng.

Output:

```text
results/demand_regime_comparison.csv
results/demand_regime_comparison.png
```

Metrics:

- economic cost.
- fill rate.
- stockout rate.
- average inventory.
- order count.

---

## P2.2 Sensitivity analysis

Script đề xuất:

```text
scripts/sensitivity_analysis.py
```

Các kịch bản ưu tiên:

### Lead time

```text
1–3 days
2–5 days
```

### Capacity

```text
capacity_cover_days = 4, 5, 6
```

### Service target

```text
0.80, 0.85, 0.90
```

### Cost ratio

Ví dụ:

```text
p/h = 5, 10, 20
```

Không cần retrain mọi sensitivity nếu mục đích chỉ test robustness của policy fixed; nhưng nếu environment/reward thay đổi đáng kể thì phải phân biệt:

```text
zero-shot evaluation
vs
retrained policy
```

Trong báo cáo không trộn hai loại này.

---

## P2.3 Ablation

Ưu tiên các flag đã có:

```yaml
normalize_reward_per_pair: true/false
normalize_adv_per_pair: true/false
use_value_norm: true/false
anneal_ent: true/false
```

Ablation chính nên tập trung:

1. reward normalization ON/OFF.
2. advantage normalization ON/OFF.
3. warehouse context ON/OFF nếu có thể triển khai sạch.

Không cần tạo quá nhiều ablation không liên quan research question.

---

# 8. State/Observation specification phải giữ đúng với code

Mỗi agent hiện có local observation khoảng 44 chiều.

Ký hiệu báo cáo:

\[
o_{i,t} =
[
I_{i,t},
IP_{i,t},
\mathbf{D}^{(7)}_{i,t},
\mathbf{P}_{i,t},
FR_{i,t},
S_i,
U_{w,t},
R^{inv}_{i,t},
G_{i,t},
\mathbf{C}_t
].
\]

Trong đó:

| Ký hiệu | Ý nghĩa |
|---|---|
| `I` | on-hand inventory |
| `IP` | inventory position = on-hand + pipeline |
| `D^(7)` | demand history 7 ngày |
| `P` | pipeline inventory theo lead-time slot |
| `FR` | rolling fill-rate signal |
| `S` | demand-scale feature |
| `U` | warehouse capacity utilization |
| `R_inv` | tỷ trọng tồn kho SKU trong warehouse |
| `G` | price/discount signal |
| `C` | calendar/day-of-week/event features |

Quan sát được chuẩn hóa về range thích hợp, phần lớn `[0,1]`.

Không tự ý thêm state variable mới nếu không có research reason.

---

# 9. Action specification

Action space:

\[
a_{i,t}\in\{0,1,2,3,4,5\}.
\]

Multipliers:

```text
[0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
```

Lượng đặt:

\[
Q_{i,t}
=
\left\lceil
m_{a_{i,t}}
\bar d_i
\bar L
\right\rceil.
\]

Với config hiện tại:

\[
\bar L = 2
\]

vì stochastic lead time nằm trong `{1,2,3}`.

Code `_build_order_table()` đang ép order levels tăng nghiêm ngặt để tránh nhiều action ánh xạ vào cùng quantity. Giữ behavior này.

---

# 10. Inventory transition để đồng bộ LaTeX

Hàng đến:

\[
R_{i,t}.
\]

Sau capacity acceptance:

\[
\tilde R_{i,t}=\rho_{w,t}R_{i,t}.
\]

Inventory trước demand:

\[
I^{pre}_{i,t}=I_{i,t}+\tilde R_{i,t}.
\]

Sales:

\[
Sales_{i,t}=\min(I^{pre}_{i,t},D_{i,t}).
\]

Stockout/lost sales:

\[
Shortage_{i,t}=\max(0,D_{i,t}-I^{pre}_{i,t}).
\]

Inventory cuối bước:

\[
I_{i,t+1}=I^{pre}_{i,t}-Sales_{i,t}.
\]

Pipeline update phải đảm bảo đơn hàng mới tới đúng lead time được sample.

---

# 11. Reward normalization

Code hiện có per-pair reward normalization. Giữ làm main model.

Nếu normalized cost setup dùng `cp_th` chung:

\[
Z_i = p\bar d_i + K.
\]

Scaled local reward:

\[
\tilde r_{i,t}
=
\frac{r_{i,t}}{Z_i}
\times RewardScale.
\]

Mục tiêu:

> tránh high-volume SKU chi phối gradient của shared policy và giúp shared critic học trên scale đồng nhất hơn.

Raw reward vẫn phải được log để diễn giải nghiệp vụ.

---

# 12. PS-IPPO theory phải khớp implementation

Không đổi algorithm family.

## Policy

\[
a_{i,t}\sim\pi_\theta(\cdot|o_{i,t}).
\]

## Value

\[
V_\phi(o_{i,t}).
\]

## TD residual

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t).
\]

## GAE

\[
\hat A_t
=
\delta_t
+\gamma\lambda\hat A_{t+1}.
\]

## PPO clipped objective

\[
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
q_t(\theta)\hat A_t,
\operatorname{clip}(q_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

Nên dùng `q_t(theta)` hoặc `r_t^ratio(theta)` cho PPO likelihood ratio để tránh nhầm với reward `r_t`.

Hyperparameters hiện tại:

```text
gamma = 0.99
gae_lambda = 0.95
clip_eps = 0.2
actor lr = 3e-4
critic lr = 1e-3
hidden_dim = 128
n_steps = 1024
ppo_epochs = 4
mini_batch_size = 16384
```

Nếu thay hyperparameter, phải log vào output và update report metadata.

---

# 13. Dữ liệu và split phải giữ rõ

M5 dùng làm **real demand source**, không phải real inventory dataset.

Không được viết “10 warehouse thật của Walmart”.

Cách diễn đạt đúng:

> Mỗi store-level demand stream trong M5 được ánh xạ thành một virtual warehouse demand node trong môi trường mô phỏng.

Split:

```text
Train:      d1 → d1050
Validation: d1051 → d1450
Test:       d1451 → d1941
```

Index code có thể zero-based, nhưng report phải diễn đạt nhất quán.

Simulation assumptions tự xây:

- initial inventory.
- stochastic lead time.
- warehouse capacity.
- holding cost.
- stockout cost.
- ordering cost.
- overflow penalty.
- ordering process.

---

# 14. Chọn 30 SKU

Không gọi là random sampling nếu preprocessing hiện không random đơn giản.

Mô tả:

> 30 SKU đại diện được lựa chọn sau khi loại các SKU có mức tiêu thụ quá thấp tại phần lớn stores, sau đó lấy mẫu phân tầng theo nhóm ngành hàng và quy mô nhu cầu.

Nếu Claude Code phát hiện implementation không đúng mô tả trên, phải:

1. báo rõ khác biệt,
2. không silently thay algorithm,
3. đề xuất patch riêng.

---

# 15. Baselines

Giữ ba baseline hiện có:

```text
EOQ
(s,S)
Newsvendor
```

Không tự thêm DQN/MAPPO chỉ để bảng dài hơn.

Có thể thêm ROP + Safety Stock sau nếu có thời gian, nhưng không phải blocker cho main retrain.

Yêu cầu fairness:

- cùng demand horizon.
- cùng lead-time stochastic seed.
- cùng warehouse capacity.
- cùng cost model.
- cùng lost-sales assumption.
- hyperparameters baseline tune trên validation.
- final numbers chỉ lấy trên test.

---

# 16. Metrics final

## Primary

```text
Economic Total Cost
Fill Rate
```

## Secondary

```text
Holding Cost
Stockout Cost
Ordering Cost
Overflow Cost
Stockout Rate
Average Inventory
Order Count/Frequency
Inventory Turnover nếu tính nhất quán được
```

## Learning diagnostics

```text
Training raw reward
Smoothed reward
Validation economic cost
Validation fill rate
Entropy
Approx KL
Value loss
Explained variance
```

---

# 17. Main experiment structure

## E1 — Main comparison

```text
PS-IPPO vs EOQ vs (s,S) vs Newsvendor
```

Trên fixed test horizon, paired stochastic seeds.

## E2 — Iso-service

So sánh economic cost với điều kiện:

\[
FillRate\ge 85\%.
\]

## E3 — Multi-seed training reliability

```text
seed = 42, 1, 2
5000 episodes/seed
```

## E4 — Demand regimes

```text
Stable / Volatile / Event
```

## E5 — Sensitivity

```text
lead time
capacity
service target
cost ratio
```

## E6 — Ablation

```text
reward normalization
advantage normalization
warehouse context nếu triển khai sạch
```

---

# 18. Preliminary results hiện tại không được coi là final

Các kết quả cũ có thể dùng làm reference/progress report nhưng sau P0 changes phải retrain.

Không overwrite chúng mà không backup.

Trước retrain final, nên archive:

```text
results_pre_fix/
checkpoints_pre_fix/
```

hoặc timestamped directory.

Claude Code phải tránh xóa kết quả cũ nếu không được yêu cầu.

---

# 19. Output folder convention đề xuất

Để phục vụ LaTeX/report reproducibility:

```text
results/
  main/
    baseline_comparison.csv
    baseline_comparison.png
    all_episodes.csv
    summary.json
    statistical_test.json
    iso_service.json
    learning_curve.png
  seeds/
    seed42/
    seed1/
    seed2/
  regimes/
  sensitivity/
  ablation/
  metadata/
    config_snapshot.yaml
    git_commit.txt   # nếu repo có git
```

Không bắt buộc refactor toàn bộ ngay nếu gây phá code lớn. Ưu tiên backward compatible.

---

# 20. Reproducibility metadata

Mỗi final run nên lưu:

```text
seed
config snapshot
n_warehouses
n_skus
n_pairs
train/val/test boundaries
algorithm name
checkpoint path
training episodes
cost parameters
lead-time range
capacity_cover_days
service target
reward normalization flags
software versions nếu thuận tiện
```

Nên xuất thành:

```text
results/run_metadata.json
```

hoặc tương đương.

---

# 21. Acceptance criteria trước khi train 5000 episodes

Claude Code chỉ coi project sẵn sàng cho final training khi tất cả điều kiện sau đạt:

### Environment

- [ ] Capacity overflow chỉ reject incoming inventory.
- [ ] Không xóa inventory cũ do capacity handling.
- [ ] Inventory không âm.
- [ ] Flow conservation tests pass.
- [ ] Pipeline arrival đúng stochastic lead time.
- [ ] Warm-start demand history hoạt động.
- [ ] Fixed full-horizon `val` và `test` hoạt động.

### Training

- [ ] True episode fill rate được log.
- [ ] Deterministic validation trả `reward`, `economic cost`, `fill rate`.
- [ ] Best checkpoint chọn theo lowest validation economic cost với `fill >= 0.85`.
- [ ] Service penalty không bị cộng vào economic cost metric.
- [ ] Training log có `eval_cost`.

### Baselines

- [ ] Baseline tune trên validation.
- [ ] Baseline dùng cùng environment/cost assumptions.
- [ ] Iso-service target không thấp hơn 0.85.

### Statistics

- [ ] Paired t-test.
- [ ] Wilcoxon signed-rank.
- [ ] Cohen's `d_z` hoặc paired effect size.
- [ ] Replications dùng paired scenario seeds.

### Reliability

- [ ] `run.py multiseed` mặc định dùng `ppo.total_episodes`.
- [ ] Không gọi 1000-episode multiseed là final reliability của 5000-episode model.

### Tests

- [ ] `python run.py test` pass.
- [ ] `python -m compileall .` pass hoặc equivalent.
- [ ] Quick smoke train/eval chạy thành công trước full train.

---

# 22. Quy trình chạy sau khi sửa code

## Bước 1 — Backup kết quả cũ

Không xóa dữ liệu cũ.

## Bước 2 — Check dependencies

```bash
python run.py check
```

## Bước 3 — Unit tests

```bash
python run.py test
```

## Bước 4 — Data preprocessing nếu cần regenerate

```bash
python run.py data
```

Không regenerate nếu raw data không có và processed data hiện tại vẫn tương thích.

## Bước 5 — Quick pipeline smoke test

```bash
python run.py all --quick
```

Phải xác nhận:

- không crash,
- cost finite,
- fill rate nằm `[0,1]`,
- overflow finite,
- checkpoint được save hợp lý.

## Bước 6 — Tune baseline validation

```bash
python run.py baseline
```

## Bước 7 — Main training

```bash
python run.py train --episodes 5000
```

## Bước 8 — Final evaluation

```bash
python run.py eval
```

## Bước 9 — Iso-service

```bash
python run.py iso
```

## Bước 10 — Multiseed

```bash
python run.py multiseed --episodes 5000
```

## Bước 11 — Summary

```bash
python run.py summary
```

Sau đó mới chạy P2 experiments.

---

# 23. Nếu compute/time hạn chế

Ưu tiên theo thứ tự:

```text
1. P0 code fixes
2. unit tests
3. quick smoke test
4. main 5000-episode seed42
5. final eval + iso-service
6. 3 full-budget seeds
7. demand regime
8. sensitivity
9. ablation
10. dashboard polishing
```

Không ưu tiên dashboard trước methodology.

---

# 24. Diagram/report requirements để hỗ trợ LaTeX

Claude Code không bắt buộc tạo LaTeX report, nhưng code output phải hỗ trợ 3 hình sau.

## 24.1 Conceptual model

```text
M5 Demand + Calendar + Price
              ↓
     Multi-Warehouse System
              ↓
        Inventory State
              ↓
          PS-IPPO
              ↓
    Replenishment Decisions
              ↓
 Inventory Dynamics + Lead Time
              ↓
 Total Cost + Fill Rate
              ↓
       Next Inventory State
```

## 24.2 RL loop

```text
Observation o_i,t
      ↓
Warehouse–SKU Agent
      ↓ Action a_i,t
Inventory Environment
      ↓
Reward + next observation
      ↺
```

## 24.3 Multi-Agent architecture

```text
Warehouse 1
 ├── SKU Agent 1 ─┐
 ├── SKU Agent 2  │
 └── ... Agent 30 │
                  │
Warehouse 2       │
 ├── ...           ├── Shared Actor πθ
                  │   Shared Critic Vφ
 ...              │
                  │
Warehouse 10      │
 └── Agent 300 ───┘

          ↓ 300 actions

Shared Multi-Warehouse Inventory Environment

Demand / Pipeline / Capacity / Inventory / Cost

          ↓ local observations + rewards

        300 agents
```

Labels cần thể hiện:

```text
No coordinator agent
No explicit inter-agent communication
Indirect interaction through shared warehouse capacity
Parameter sharing
```

---

# 25. LaTeX-ready theoretical formulation

Claude Code nên giữ tên biến/code đủ rõ để map vào các ký hiệu sau.

## Multi-agent formulation

\[
\mathcal{G}
=
\left\langle
\mathcal{N},
\mathcal{S},
\{\mathcal{O}_i\}_{i=1}^{N},
\{\mathcal{A}_i\}_{i=1}^{N},
P,
\{R_i\}_{i=1}^{N},
\gamma
\right\rangle.
\]

với:

\[
N=300.
\]

## Service metric

\[
FillRate
=
1-
\frac{\sum_t Shortage_t}
{\sum_t Demand_t}.
\]

## Cost objective

\[
\min_{\pi}
\mathbb{E}
\left[
\sum_t
(C^H_t+C^S_t+C^O_t+C^V_t)
\right]
\]

subject to:

\[
FillRate\ge0.85.
\]

---

# 26. Các câu hỏi GVHD mà implementation phải trả lời được

Sau khi sửa code, hệ thống phải đủ rõ để trả lời:

### “I trong IPPO là gì?”

Independent — mỗi agent ra quyết định từ local observation và không dùng centralized critic/joint action trong execution.

### “Sharing network rồi sao còn Independent?”

Các agent dùng chung parameters nhưng có observation/action riêng. Parameter sharing khác với việc hợp nhất agent.

### “Có bao nhiêu agent?”

300 warehouse–SKU agents.

### “Mỗi agent làm gì?”

Quyết định replenishment quantity của một SKU tại một warehouse.

### “Có coordinator không?”

Không.

### “Các agent tương tác thế nào?”

Gián tiếp qua shared warehouse capacity trong cùng warehouse; không giao tiếp message trực tiếp.

### “Tại sao bài toán là RL?”

Vì là sequential decision problem; action đặt hàng hôm nay ảnh hưởng inventory và cost tương lai qua stochastic lead time, demand uncertainty và capacity constraints.

### “RL có chắc thắng EOQ không?”

Không. So sánh được quyết định bằng thực nghiệm; đặc biệt phải xem cost–service tradeoff và demand regime.

---

# 27. MCP

Không tích hợp MCP vào PS-IPPO hiện tại.

Chỉ ghi note/documentation:

> MCP được khảo sát ở góc độ context/task/tool orchestration cho lớp ứng dụng/agentic system mở rộng. MCP không phải coordinator của 300 RL agents và không tham gia trực tiếp vào PPO training.

Không thêm dependency MCP nếu không có yêu cầu sau.

---

# 28. Dashboard/Simulation

Không phải blocker của final training.

Sau khi model ổn định, dashboard có thể hiển thị theo ngày:

```text
Date
Warehouse
SKU
Demand
On-hand inventory
Pipeline incoming
Order action
Order quantity
Stockout
Fill rate
Holding cost
Stockout cost
Ordering cost
Overflow
Reward
```

Không refactor dashboard trước khi P0/P1 hoàn thành.

---

# 29. Claude Code deliverables

Sau khi xử lý file này, Claude Code phải trả về:

## A. Code changes

Danh sách file đã sửa và tóm tắt mỗi thay đổi.

## B. Tests

Kết quả:

```text
pytest
compileall
quick smoke pipeline
```

## C. Behavior validation

Xác nhận cụ thể:

```text
capacity bug fixed
true fill rate implemented
checkpoint feasible selection implemented
validation baseline tuning implemented
paired statistics implemented
fixed eval horizon implemented
warm-start history implemented
multiseed budget aligned
```

## D. Không train full ngay nếu test fail

Nếu unit/smoke test fail, dừng ở mức code fix và báo lỗi.

## E. Không tự ý thay đổi research scope

Không thay IPPO bằng thuật toán khác.

## F. Sau khi code pass

Đề xuất exact commands để user chạy full training.

---

# 30. Change management rules

1. Ưu tiên patch nhỏ, backward compatible.
2. Không xóa kết quả/checkpoint cũ.
3. Không đổi tên output file đang được dashboard/script khác dùng nếu không cập nhật toàn bộ references.
4. Nếu cần đổi JSON schema, giữ backward-compatible keys khi hợp lý.
5. Mọi thay đổi ảnh hưởng methodology phải được ghi vào `CHANGELOG_KLTN.md` hoặc tương đương.
6. Không silently sửa công thức research mà không ghi lại.
7. Không dùng test data để chọn hyperparameter/checkpoint.
8. Không report `service penalty` như operating cost thực.
9. Không gọi M5 store là warehouse thật; dùng `virtual warehouse demand node`.
10. Không gọi bài toán là multi-echelon.

---

# 31. Đề xuất thứ tự commit

Nếu repo có git, chia commit:

```text
fix(env): reject overflow only from incoming inventory
fix(train): use true fill rate and constrained checkpoint selection
fix(eval): paired statistics and fixed evaluation horizon
fix(baselines): tune on validation with matched environment
fix(multiseed): align training budget
feat(eval): demand regime experiment
feat(report): reproducibility metadata and result organization
```

Không bắt buộc tên commit đúng y như trên.

---

# 32. Definition of Done cho bản code dùng viết KLTN

Bản code được coi là đủ để train và viết Chương 4 khi:

- environment tests pass.
- capacity logic đúng.
- train/val/test separation đúng.
- best model không dùng test data.
- business metrics đúng.
- economic objective và reward shaping tách rõ.
- baseline comparison fair.
- statistical comparison paired.
- 3 seeds chạy cùng budget.
- raw outputs đủ tái tạo bảng/figure.
- report có thể giải thích agent/state/action/reward/transition từ code mà không mâu thuẫn.

---

# 33. Ưu tiên thực thi ngay

Claude Code hãy thực hiện theo thứ tự sau:

```text
PHASE 1 — Audit
1. Đọc config.yaml.
2. Đọc env/inventory_env.py.
3. Đọc scripts/train.py.
4. Đọc scripts/tune_baselines.py.
5. Đọc scripts/evaluate.py.
6. Đọc scripts/iso_service.py.
7. Đọc run.py.
8. Đọc tests/test_env.py.
9. Đọc agents/ppo_agent.py và rollout_buffer.py để đảm bảo thuật ngữ PS-IPPO đúng implementation.

PHASE 2 — P0 fixes
10. Fix overflow.
11. Add regression tests.
12. Set main normalized cost config.
13. Fix true fill rate.
14. Fix deterministic eval economic cost.
15. Fix checkpoint selection.
16. Add eval_cost logging.

PHASE 3 — P1 fixes
17. Tune baselines on validation.
18. Sync baseline environment inputs.
19. Fix iso-service target and validation tuning.
20. Fix paired statistics.
21. Add fixed full-horizon val/test mode.
22. Warm-start demand history.
23. Align multiseed training budget.

PHASE 4 — Verify
24. Run compileall.
25. Run pytest.
26. Run quick smoke train/eval/iso.
27. Inspect generated CSV/JSON for NaN/invalid fill/cost.

PHASE 5 — Report readiness
28. Produce concise CHANGELOG.
29. Produce final command list for 5000-episode training.
30. Do not start expensive full training automatically unless explicitly requested.
```

---

# 34. Quan trọng nhất

**Không tối ưu code để “ép IPPO thắng baseline”.**

Mục tiêu là experimental design hợp lệ.

Một kết quả hoàn toàn hợp lệ có thể là:

- classical policy rẻ hơn trong demand ổn định;
- PS-IPPO đạt service tốt hơn;
- PS-IPPO có lợi thế trong demand biến động;
- hoặc PS-IPPO không vượt baseline ở một số cấu hình.

Kết luận phải xuất phát từ test data, không từ mong muốn của đề tài.

---

# END — CLAUDE CODE SHOULD TREAT THIS FILE AS THE CURRENT IMPLEMENTATION PLAN
