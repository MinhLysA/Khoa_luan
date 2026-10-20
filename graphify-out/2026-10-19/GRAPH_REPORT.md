# Graph Report - Khoa_luan  (2026-10-19)

## Corpus Check
- 107 files · ~335,488 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 72 file(s) not represented in the graph (top: .pth 27, .tex 17, .csv 12)

## Summary
- 522 nodes · 1245 edges · 16 communities (15 shown, 1 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 94 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Chạy chính sách & baseline
- Tác tử PPO
- Kiểm thử môi trường
- Điều phối run.py & demo data
- Mô phỏng & sốc cầu
- Campaign & kết quả sốc
- Cấu hình ablation
- MCP server
- Lõi demo (core.py)
- Nạp dữ liệu & cache demo
- Trang demo & định dạng số
- Tổng hợp kết quả
- Trang đơn đặt hàng & UI
- Đọc results & bảng cặp
- Tiền xử lý dữ liệu M5

## God Nodes (most connected - your core abstractions)
1. `MultiWarehouseInventoryEnv` - 42 edges
2. `make_env()` - 42 edges
3. `train()` - 26 edges
4. `PPOAgent` - 25 edges
5. `RolloutBuffer` - 16 edges
6. `SsPolicy` - 16 edges
7. `NewsvendorPolicy` - 16 edges
8. `make_policies()` - 16 edges
9. `EOQPolicy` - 14 edges
10. `build_env_state()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Thesis scope and theory` --references--> `TIEN_DO progress tracker`  [EXTRACTED]
  rl_inventory/pham_vi_khoa_luan.md → TIEN_DO.md
- `README run guide` --references--> `TIEN_DO progress tracker`  [EXTRACTED]
  rl_inventory/README.md → TIEN_DO.md
- `include_price_signal 44-dim observation` --references--> `config.yaml base config`  [EXTRACTED]
  TIEN_DO.md → rl_inventory/config.yaml
- `MDP formulation of multi-warehouse inventory` --conceptually_related_to--> `PS-IPPO (parameter-sharing IPPO)`  [INFERRED]
  rl_inventory/van_dap.md → TIEN_DO.md
- `Colab experiment workflow` --implements--> `Final training campaign (3 main + 16 ablation/hold-out)`  [INFERRED]
  rl_inventory/README.md → TIEN_DO.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Hold-out generalization experiment** — rl_inventory_configs_holdout_train_setting, rl_inventory_configs_holdout_eval_setting, rl_inventory_pham_vi_khoa_luan_rq2 [INFERRED 0.85]
- **RQ3 reward design ablations** — rl_inventory_configs_ablation_global_reward_setting, rl_inventory_configs_ablation_q3_no_reward_norm_setting, rl_inventory_pham_vi_khoa_luan_rq3 [INFERRED 0.85]

## Communities (16 total, 1 thin omitted)

### Community 0 - "Chạy chính sách & baseline"
Cohesion: 0.07
Nodes (24): load_all_policies(), load_baseline_params(), load_baselines(), BasePolicy, build_env_state(), demand_group_index(), EOQPolicy, expand_group_params() (+16 more)

### Community 1 - "Tác tử PPO"
Cohesion: 0.05
Nodes (10): _mlp(), PPOAgent, SharedActorCriticNetwork, ValueNormalizer, RolloutBuffer, load_ippo(), load_data(), train() (+2 more)

### Community 2 - "Kiểm thử môi trường"
Cohesion: 0.07
Nodes (41): make_env(), test_ba_mien_train_val_test_khong_chong_lan(), test_bang_dat_hang_tang_nghiem_ngat(), test_bao_toan_hang_hoa(), test_chi_phi_dat_hang_theo_tung_cap(), test_config_ablation_dong_bo_voi_config_chinh(), test_cong_thuc_chi_phi_dung(), test_cp_th_pair_mac_dinh_la_hang_so_khi_tat_gia_that() (+33 more)

### Community 3 - "Điều phối run.py & demo data"
Cohesion: 0.07
Nodes (29): main(), sh(), main(), rollout(), load_agent(), load_config(), load_data(), make_env() (+21 more)

### Community 4 - "Mô phỏng & sốc cầu"
Cohesion: 0.08
Nodes (12): apply_demand_shock(), current_obs(), make_env(), make_scenario_env(), run_episode(), run_parallel(), summarize(), factory() (+4 more)

### Community 5 - "Campaign & kết quả sốc"
Cohesion: 0.10
Nodes (22): shock_results(), acquire(), ck(), done(), evaluate_all(), latest_checkpoint(), _lock(), lock_owner() (+14 more)

### Community 6 - "Cấu hình ablation"
Cohesion: 0.06
Nodes (25): config.yaml base config, Ablation: drop 27 calendar features, Ablation: drop warehouse-level signals, Ablation: add event lookahead features, Ablation: global (team) reward, Ablation: no per-pair reward normalization, Ablation: demand-scaled SLA penalty, Ablation: shared actor/critic trunk (v1) (+17 more)

### Community 7 - "MCP server"
Cohesion: 0.11
Nodes (16): default_checkpoint(), load_env_bundle(), canh_bao_thieu_hang(), _ctx(), danh_sach_kho_va_mat_hang(), de_xuat_dat_hang(), ket_qua_tong_hop(), _kpi() (+8 more)

### Community 8 - "Lõi demo (core.py)"
Cohesion: 0.12
Nodes (12): chart(), obs_labels(), shock_metrics(), style_fig(), test_app_khong_import_train(), test_chi_so_soc_ngay_ve_lai(), test_demo_data_nap_duoc(), test_mo_phong_truc_tiep_khop_rollout_da_chay_san() (+4 more)

### Community 9 - "Nạp dữ liệu & cache demo"
Cohesion: 0.19
Nodes (7): base_env(), load_demo(), load_policies(), make_env(), rollout(), shock_rollout(), state_at()

### Community 10 - "Trang demo & định dạng số"
Cohesion: 0.15
Nodes (8): num(), pct(), dau(), diem(), khung(), la_tot(), noi(), pc()

### Community 11 - "Tổng hợp kết quả"
Cohesion: 0.28
Nodes (12): _fmt_money(), _load_json(), main(), section_baselines(), section_env(), section_eval(), section_files(), section_iso() (+4 more)

### Community 12 - "Trang đơn đặt hàng & UI"
Cohesion: 0.14
Nodes (3): nguon(), no_data(), tieu_de()

### Community 13 - "Đọc results & bảng cặp"
Cohesion: 0.18
Nodes (4): pairs(), read_results(), summary(), mep()

### Community 14 - "Tiền xử lý dữ liệu M5"
Cohesion: 0.22
Nodes (5): generate_calendar_features(), generate_synthetic_data(), load_price_data(), main(), preprocess_m5_data()

## Knowledge Gaps
- **11 isolated node(s):** `Baselines EOQ, (s,S), Newsvendor`, `M5 Walmart data (10 stores x 30 SKU)`, `RQ1 PS-IPPO vs baselines at fill rate >= 85%`, `MDP formulation of multi-warehouse inventory`, `MCP prototype and simulation dashboard` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 181 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MultiWarehouseInventoryEnv` connect `Mô phỏng & sốc cầu` to `Chạy chính sách & baseline`, `Tác tử PPO`, `Kiểm thử môi trường`, `Điều phối run.py & demo data`, `Lõi demo (core.py)`, `Nạp dữ liệu & cache demo`, `Tổng hợp kết quả`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `MultiWarehouseInventoryEnv` (e.g. with `load_all_policies()` and `load_baselines()`) actually correct?**
  _`MultiWarehouseInventoryEnv` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Baselines EOQ, (s,S), Newsvendor`, `M5 Walmart data (10 stores x 30 SKU)`, `RQ1 PS-IPPO vs baselines at fill rate >= 85%` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Chạy chính sách & baseline` be split into smaller, more focused modules?**
  _Cohesion score 0.06559356136820925 - nodes in this community are weakly interconnected._
- **Why does `train()` connect `Tác tử PPO` to `Chạy chính sách & baseline`, `Kiểm thử môi trường`, `Điều phối run.py & demo data`, `Mô phỏng & sốc cầu`, `Campaign & kết quả sốc`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `train()` (e.g. with `.load()` and `.save()`) actually correct?**
  _`train()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Should `Tác tử PPO` be split into smaller, more focused modules?**
  _Cohesion score 0.05427547363031234 - nodes in this community are weakly interconnected._