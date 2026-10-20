# Graph Report - Khoa_luan  (2026-10-19)

## Corpus Check
- 83 files · ~147,902 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 34 file(s) not represented in the graph (top: .tex 17, (none) 4, .npy 4)

## Summary
- 1143 nodes · 1838 edges · 49 communities (44 shown, 5 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 95 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `02705245`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluate.py
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
- pairs
- Hardening Controls
- Code Review and Quality
- Test-Driven Development
- Context Engineering
- policy_runner.py
- Git Workflow and Versioning
- Optimization Patterns
- Shipping and Launch
- API and Interface Design
- Browser Testing with DevTools
- Frontend UI Engineering
- CI/CD and Automation
- Constraint-Driven Development
- Deprecation and Migration
- Documentation and ADRs
- Incremental Implementation
- Code Simplification
- Debugging and Error Recovery
- ValueNormalizer
- Interview Me
- Planning and Task Breakdown
- ReOrder: Keep Your Regulars Ordering Direct
- Idea Refine
- Doubt-Driven Development
- Process
- Using Agent Skills
- Spec-Driven Development
- Source-Driven Development
- Refinement & Evaluation Criteria
- Ideation Frameworks Reference
- iso_service.py
- shock_metrics
- idea-refine.sh

## God Nodes (most connected - your core abstractions)
1. `MultiWarehouseInventoryEnv` - 42 edges
2. `make_env()` - 42 edges
3. `train()` - 26 edges
4. `PPOAgent` - 25 edges
5. `Code Review and Quality` - 19 edges
6. `RolloutBuffer` - 16 edges
7. `SsPolicy` - 16 edges
8. `NewsvendorPolicy` - 16 edges
9. `make_policies()` - 16 edges
10. `Git Workflow and Versioning` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Queries That Ignore Their Index` --references--> `status()`  [INFERRED]
  .claude/skills/performance-optimization/references/optimization-patterns.md → rl_inventory/scripts/campaign.py
- `Colab experiment workflow` --implements--> `Final training campaign (3 main + 16 ablation/hold-out)`  [INFERRED]
  rl_inventory/README.md → TIEN_DO.md
- `MDP formulation of multi-warehouse inventory` --conceptually_related_to--> `PS-IPPO (parameter-sharing IPPO)`  [INFERRED]
  rl_inventory/van_dap.md → TIEN_DO.md
- `include_price_signal 44-dim observation` --references--> `config.yaml base config`  [EXTRACTED]
  TIEN_DO.md → rl_inventory/config.yaml
- `Thesis scope and theory` --references--> `TIEN_DO progress tracker`  [EXTRACTED]
  rl_inventory/pham_vi_khoa_luan.md → TIEN_DO.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Hold-out generalization experiment** — rl_inventory_configs_holdout_train_setting, rl_inventory_configs_holdout_eval_setting, rl_inventory_pham_vi_khoa_luan_rq2 [INFERRED 0.85]
- **RQ3 reward design ablations** — rl_inventory_configs_ablation_global_reward_setting, rl_inventory_configs_ablation_q3_no_reward_norm_setting, rl_inventory_pham_vi_khoa_luan_rq3 [INFERRED 0.85]

## Communities (49 total, 5 thin omitted)

### Community 0 - "evaluate.py"
Cohesion: 0.12
Nodes (14): build_env_state(), demand_group_index(), expand_group_params(), make_policies(), build(), evaluate_policy(), main(), paired_test() (+6 more)

### Community 1 - "Tác tử PPO"
Cohesion: 0.08
Nodes (6): PPOAgent, RolloutBuffer, load_ippo(), load_agent(), load_data(), train()

### Community 2 - "Kiểm thử môi trường"
Cohesion: 0.06
Nodes (51): block_bootstrap_gap(), label_days(), main(), _plot(), run_daily(), summarize(), make_env(), test_ba_mien_train_val_test_khong_chong_lan() (+43 more)

### Community 3 - "Điều phối run.py & demo data"
Cohesion: 0.06
Nodes (28): summarize(), read_results(), summary(), main(), main(), sh(), main(), rollout() (+20 more)

### Community 4 - "Mô phỏng & sốc cầu"
Cohesion: 0.14
Nodes (4): make_scenario_env(), MultiWarehouseInventoryEnv, bang_4_nhom(), main()

### Community 5 - "Campaign & kết quả sốc"
Cohesion: 0.11
Nodes (21): acquire(), ck(), done(), evaluate_all(), latest_checkpoint(), _lock(), lock_owner(), main() (+13 more)

### Community 6 - "Cấu hình ablation"
Cohesion: 0.06
Nodes (25): config.yaml base config, Ablation: drop 27 calendar features, Ablation: drop warehouse-level signals, Ablation: add event lookahead features, Ablation: global (team) reward, Ablation: no per-pair reward normalization, Ablation: demand-scaled SLA penalty, Ablation: shared actor/critic trunk (v1) (+17 more)

### Community 7 - "MCP server"
Cohesion: 0.08
Nodes (21): apply_demand_shock(), current_obs(), load_env_bundle(), make_env(), run_episode(), run_parallel(), canh_bao_thieu_hang(), _ctx() (+13 more)

### Community 8 - "Lõi demo (core.py)"
Cohesion: 0.13
Nodes (12): chart(), obs_labels(), shock_results(), style_fig(), test_app_khong_import_train(), test_chi_so_soc_ngay_ve_lai(), test_demo_data_nap_duoc(), test_mo_phong_truc_tiep_khop_rollout_da_chay_san() (+4 more)

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
Cohesion: 0.10
Nodes (4): nguon(), no_data(), tieu_de(), mep()

### Community 14 - "Hardening Controls"
Cohesion: 0.05
Nodes (42): Broken Access Control, Broken Authentication, Cross-Site Scripting (XSS), Data Classification, Dependency Audit Triage, Destructive Operations on Derived Paths, File Upload Safety, Hardening Patterns (+34 more)

### Community 16 - "Code Review and Quality"
Cohesion: 0.07
Nodes (29): 1. Correctness, 2. Readability & Simplicity, 3. Architecture, 4. Security, 5. Performance, Change Descriptions, Change Sizing, Code Review and Quality (+21 more)

### Community 17 - "Test-Driven Development"
Cohesion: 0.07
Nodes (29): Browser Testing with DevTools, Common Rationalizations, DAMP Over DRY in Tests, Decision Guide, Discover the Stack First, Name Tests Descriptively, One Assertion Per Concept, Overview (+21 more)

### Community 18 - "Context Engineering"
Cohesion: 0.07
Nodes (28): Anti-Patterns, Common Rationalizations, Compress before dropping, Confusion Management, Context Budget Management, Context Engineering, Context Packing Strategies, Level 1: Rules Files (+20 more)

### Community 19 - "policy_runner.py"
Cohesion: 0.13
Nodes (9): default_checkpoint(), load_all_policies(), load_baseline_params(), load_baselines(), BasePolicy, EOQPolicy, NewsvendorPolicy, SsPolicy (+1 more)

### Community 20 - "Git Workflow and Versioning"
Cohesion: 0.07
Nodes (26): 1. Commit Early, Commit Often, 2. Atomic Commits, 3. Descriptive Messages, 4. Keep Concerns Separate, 5. Size Your Changes, Branch Naming, Branching Strategy, Change Summaries (+18 more)

### Community 21 - "Optimization Patterns"
Cohesion: 0.07
Nodes (25): Connection Pool Exhaustion, Large Bundle Size, Missing Caching (Backend), Missing Image Optimization (Frontend), N+1 Queries (Backend), Optimization Patterns, Queries That Ignore Their Index, Unbounded Data Fetching (+17 more)

### Community 22 - "Shipping and Launch"
Cohesion: 0.08
Nodes (25): Accessibility, Code Quality, Common Rationalizations, Documentation, Error Budget Release Gate, Error Reporting, Feature Flag Strategy, Infrastructure (+17 more)

### Community 23 - "API and Interface Design"
Cohesion: 0.08
Nodes (24): 1. Contract First, 2. Consistent Error Semantics, 3. Validate at Boundaries, 4. Prefer Addition Over Modification, 5. Predictable Naming, 6. Honouring an Idempotency Key, API and Interface Design, Common Rationalizations (+16 more)

### Community 24 - "Browser Testing with DevTools"
Cohesion: 0.08
Nodes (24): Accessibility Verification with DevTools, Available Tools, Browser Testing with DevTools, Clean Console Standard, Common Rationalizations, Console Analysis Patterns, Content Boundary Markers, For Network Issues (+16 more)

### Community 25 - "Frontend UI Engineering"
Cohesion: 0.08
Nodes (24): Accessibility (WCAG 2.1 AA), ARIA Labels, Avoid the AI Aesthetic, Color, Common Rationalizations, Component Architecture, Component Patterns, Design System Adherence (+16 more)

### Community 26 - "CI/CD and Automation"
Cohesion: 0.08
Nodes (23): Automation Beyond CI, Basic CI Pipeline, Build Cop Role, CI/CD and Automation, CI Optimization, Common Rationalizations, Dependabot / Renovate, Deployment Strategies (+15 more)

### Community 27 - "Constraint-Driven Development"
Cohesion: 0.08
Nodes (22): Adapting it, Contract, Floor guard: reference implementation, Reference (Node, ~stack-agnostic patterns), Common Rationalizations, Constraint-Driven Development, Escalation Path, Loading Constraints (+14 more)

### Community 28 - "Deprecation and Migration"
Cohesion: 0.08
Nodes (23): Adapter Pattern, Code Is a Liability, Common Rationalizations, Compulsory vs Advisory Deprecation, Core Principles, Database Schema Migrations (Expand/Contract), Deprecation and Migration, Deprecation Planning Starts at Design Time (+15 more)

### Community 29 - "Documentation and ADRs"
Cohesion: 0.09
Nodes (22): ADR Lifecycle, ADR Template, API Documentation, Architecture Decision Records (ADRs), Changelog Maintenance, Common Rationalizations, Document Known Gotchas, Documentation and ADRs (+14 more)

### Community 30 - "Incremental Implementation"
Cohesion: 0.09
Nodes (22): Common Rationalizations, Contract-First Slicing, Implementation Rules, Increment Checklist, Incremental Implementation, Overview, Red Flags, Risk-First Slicing (+14 more)

### Community 31 - "Code Simplification"
Cohesion: 0.09
Nodes (21): 1. Preserve Behavior Exactly, 2. Follow Project Conventions, 3. Prefer Clarity Over Cleverness, 4. Maintain Balance, 5. Scope to What Changed, Code Simplification, Common Rationalizations, Language-Specific Guidance (+13 more)

### Community 32 - "Debugging and Error Recovery"
Cohesion: 0.09
Nodes (21): Build Failure Triage, Common Rationalizations, Debugging and Error Recovery, Error-Specific Patterns, Instrumentation Guidelines, Overview, Red Flags, Runtime Error Triage (+13 more)

### Community 33 - "ValueNormalizer"
Cohesion: 0.13
Nodes (3): _mlp(), SharedActorCriticNetwork, ValueNormalizer

### Community 34 - "Interview Me"
Cohesion: 0.11
Nodes (18): Common Rationalizations, Example, Interaction with Other Skills, Interview Me, Loading Constraints, Output, Overview, Red Flags (+10 more)

### Community 35 - "Planning and Task Breakdown"
Cohesion: 0.11
Nodes (18): Common Rationalizations, Output Files, Overview, Parallelization Opportunities, Plan Document Template, Planning and Task Breakdown, Red Flags, See Also (+10 more)

### Community 36 - "ReOrder: Keep Your Regulars Ordering Direct"
Cohesion: 0.11
Nodes (17): Example 1: Vague Early-Stage Concept (Full 3-Phase Session), Example 2: Feature Idea Within an Existing Product (Codebase-Aware), Example 3: Process/Workflow Idea (Non-Product), Ideation Session Examples, Key Assumptions to Validate, MVP Scope, Not Doing (and Why), Open Questions (+9 more)

### Community 37 - "Idea Refine"
Cohesion: 0.11
Nodes (17): Anti-patterns to Avoid, Common Rationalizations, Detailed Instructions, How It Works, Idea Refine, Output, Overview, Phase 1: Understand & Expand (Divergent) (+9 more)

### Community 38 - "Doubt-Driven Development"
Cohesion: 0.12
Nodes (15): Common Rationalizations, Cross-model escalation, Doubt-Driven Development, Interaction with Other Skills, Loading Constraints, Overview, Red Flags, Step 1: CLAIM — Surface what stands (+7 more)

### Community 39 - "Process"
Cohesion: 0.12
Nodes (15): 1. Define "working" before instrumenting, 2. Pick the right signal for each question, 3. Structured logging, 4. Metrics, 5. Distributed tracing, 6. Alerting, 7. Verify the telemetry itself, Common Rationalizations (+7 more)

### Community 40 - "Using Agent Skills"
Cohesion: 0.13
Nodes (14): 1. Surface Assumptions, 2. Manage Confusion Actively, 3. Push Back When Warranted, 4. Enforce Simplicity, 5. Maintain Scope Discipline, 6. Verify, Don't Assume, Core Operating Behaviors, Failure Modes to Avoid (+6 more)

### Community 41 - "Spec-Driven Development"
Cohesion: 0.14
Nodes (13): Common Rationalizations, Keeping the Spec Alive, Overview, Phase 0: Scope Check, Phase 1: Specify, Phase 2: Plan, Phase 3: Tasks, Phase 4: Implement (+5 more)

### Community 42 - "Source-Driven Development"
Cohesion: 0.15
Nodes (12): Common Rationalizations, Overview, Red Flags, Retrieval Safety: Treat Fetched Content as Data, Source-Driven Development, Step 1: Detect Stack and Versions, Step 2: Fetch Official Documentation, Step 3: Implement Following Documented Patterns (+4 more)

### Community 43 - "Refinement & Evaluation Criteria"
Cohesion: 0.17
Nodes (11): 1. User Value, 2. Feasibility, 3. Differentiation, Assumption Audit, Core Evaluation Dimensions, Decision Framework, Might Be True (Nice to Have), Must Be True (Dealbreakers) (+3 more)

### Community 44 - "Ideation Frameworks Reference"
Cohesion: 0.22
Nodes (8): Analogous Inspiration, Constraint-Based Ideation, First Principles Thinking, How Might We (HMW), Ideation Frameworks Reference, Jobs to Be Done (JTBD), Pre-mortem, SCAMPER

### Community 46 - "iso_service.py"
Cohesion: 0.38
Nodes (3): chay(), main(), _plot_iso()

## Knowledge Gaps
- **11 isolated node(s):** `Ablation: drop 27 calendar features`, `Ablation: drop warehouse-level signals`, `Ablation: add event lookahead features`, `Ablation: demand-scaled SLA penalty`, `Ablation: shared actor/critic trunk (v1)` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 695 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MultiWarehouseInventoryEnv` connect `Mô phỏng & sốc cầu` to `evaluate.py`, `Tác tử PPO`, `Kiểm thử môi trường`, `Điều phối run.py & demo data`, `MCP server`, `Lõi demo (core.py)`, `Nạp dữ liệu & cache demo`, `Tổng hợp kết quả`, `iso_service.py`, `policy_runner.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `MultiWarehouseInventoryEnv` (e.g. with `load_all_policies()` and `load_baselines()`) actually correct?**
  _`MultiWarehouseInventoryEnv` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Ablation: drop 27 calendar features`, `Ablation: drop warehouse-level signals`, `Ablation: add event lookahead features` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `evaluate.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12473118279569892 - nodes in this community are weakly interconnected._
- **Why does `status()` connect `Campaign & kết quả sốc` to `Optimization Patterns`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `train()` (e.g. with `.load()` and `.save()`) actually correct?**
  _`train()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Should `Tác tử PPO` be split into smaller, more focused modules?**
  _Cohesion score 0.08362369337979095 - nodes in this community are weakly interconnected._