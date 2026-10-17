"""Trang 6 - Về mô hình (tĩnh, để thuyết trình). Số lấy từ cấu hình đã đóng gói và results/."""
import plotly.graph_objects as go
import streamlit as st

import core

meta, _ = core.load_demo()
env = core.base_env()
ppo = meta["ppo"]
n, d, h, k = env.n_pairs, env.obs_per_pair, int(ppo["hidden_dim"]), env.n_action_levels
mau = int(ppo["n_steps"]) * n

core.tieu_de("Về mô hình · IPPO", "Về mô hình IPPO trong mô phỏng",
             "Một mạng dùng chung cho mọi cặp (kho, SKU).")

st.subheader("Một mạng dùng chung cho mọi cặp")
o = [(f"{n} cặp (kho, SKU)", f"{env.n_warehouses} kho × {env.n_skus} SKU"),
     (f"Batch ({n}, {d})", f"mỗi cặp một vector quan sát {d} chiều"),
     ("1 mạng Actor–Critic dùng chung",
      f"Actor {d}→{h}→{h}→{k} · Critic {d}→{h}→{h}→1"),
     (f"{n} hành động", f"mỗi cặp chọn 1 trong {k} mức đặt hàng")]
cols = st.columns([3, 1, 3, 1, 3, 1, 3])
for i, (ten, mo_ta) in enumerate(o):
    cols[2 * i].markdown(f'<div class="box"><b>{ten}</b><br><span style="color:{core.MUTED}">'
                         f'{mo_ta}</span></div>', unsafe_allow_html=True)
    if i < len(o) - 1:
        cols[2 * i + 1].markdown('<div style="text-align:center;font-size:1.6rem">→</div>',
                                 unsafe_allow_html=True)

st.subheader("Hai kênh ghép nối giữa các cặp")
g1, g2 = st.columns(2)
g1.markdown(f"**1. Chia sẻ gradient.** Cả {n} cặp dùng chung một bộ tham số; mỗi lần cập nhật, "
            f"gradient từ mẫu của mọi cặp cùng đổ vào một mạng, nên kinh nghiệm của cặp này "
            f"được dùng cho cặp khác.")
g2.markdown("**2. Tín hiệu util/share.** Quan sát của mỗi cặp có mức sử dụng sức chứa của kho "
            "(util) và tỷ trọng tồn của cặp trong kho (share). Các cặp cùng kho chia nhau một "
            "sức chứa, nên hàng mà cặp khác đặt về hiện ra trong quan sát của cặp này.")

st.subheader("Một lần cập nhật gồm 5 bước")
chuan_hoa = ", chuẩn hóa advantage theo từng cặp" if ppo.get("normalize_adv_per_pair") else ""
st.markdown(f"""
1. **Rollout** {core.num(ppo['n_steps'])} bước × {n} cặp với chính sách hiện tại.
2. **GAE độc lập theo cặp** (γ = {core.num(ppo['gamma'], 2)}, λ = {core.num(ppo['gae_lambda'], 2)}){chuan_hoa}.
3. **Làm phẳng** thành {core.num(ppo['n_steps'])} × {n} = {core.num(mau)} mẫu.
4. **PPO** {ppo['ppo_epochs']} epoch, mini-batch {core.num(ppo['mini_batch_size'])}, clip ε = {core.num(ppo['clip_eps'], 1)}.
5. **Đánh giá** trên miền validation mỗi {ppo['eval_every_episodes']} episode để chọn best_model.
""")

with st.container(border=True, key="the_reward"):
    st.subheader("Đường cong phần thưởng khi huấn luyện")
    log = core.read_results(f"train_log_{meta['results_tag']}.csv")
    if log is None:
        core.no_data(f"results/train_log_{meta['results_tag']}.csv")
    else:
        fig = go.Figure([
            go.Scatter(x=log["episode"], y=log["reward_raw"], name="Từng episode",
                       line=dict(color=core.GREY, width=1), opacity=0.5),
            go.Scatter(x=log["episode"], y=log["reward_smooth"], name="Trung bình trượt",
                       line=dict(color=core.TEAL, width=2.5))])
        fig.update_xaxes(title="Episode")
        fig.update_yaxes(title="Tổng phần thưởng (âm chi phí)")
        core.chart(fig, 420)
core.nguon(f"results/train_log_{meta['results_tag']}.csv (chỉ đọc)", "cấu hình PPO trong app/demo_data/meta.json")
