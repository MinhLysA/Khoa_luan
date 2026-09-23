"""
app/mcp_server.py
=================
Prototype MCP (Model Context Protocol): dua engine IPPO thanh cac CONG CU
(tool) ma mot LLM (Claude, ...) goi duoc. Minh hoa luong dieu phoi:

    Nguoi dung dat cau hoi (Task)
      -> LLM (host) chon tool phu hop (Action)
      -> tool lay ngu canh tu moi truong kho (Context) va goi IPPO/baseline
      -> ket qua tra ve LLM de tong hop cau tra loi.

IPPO van la noi ra quyet dinh dinh luong; MCP chi lo giao tiep va dieu phoi.

Chay (stdio, dung cho Claude Desktop / Claude Code):
    python app/mcp_server.py
Tu kiem tra khong can LLM:
    python app/mcp_server.py --selftest

Khai bao cho Claude Code:
    claude mcp add kho-ippo -- python "<duong dan>/rl_inventory/app/mcp_server.py"
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend import simulator, policy_runner  # noqa: E402

try:                                      # MCP Python SDK v1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:                       # SDK v2 doi ten
    from mcp.server.mcpserver import MCPServer as _Server

mcp = _Server("kho-ippo")


# --------------------------------------------------------------------------- #
# Ngu canh dung chung
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _ctx():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    bundle = simulator.load_env_bundle()
    meta = bundle["meta"]
    return cfg, bundle, meta.get("stores"), meta.get("top_items")


def _policies(env):
    cfg, *_ = _ctx()
    ck = policy_runner.default_checkpoint(ROOT / "checkpoints")
    return policy_runner.load_all_policies(env, str(ck) if ck else None,
                                           cfg["ppo"], ROOT / "results")


def _run_to_day(ngay: int, chinh_sach: str = "IPPO", seed: int = 1000):
    """Mo phong mien test tu ngay dau (ngay 1450 cua M5) toi `ngay` ngay sau,
    theo mot chinh sach. Tra ve env o dung trang thai ngay do."""
    cfg, bundle, *_ = _ctx()
    env = simulator.make_env(cfg, bundle, mode="test")
    pols = _policies(env)
    if chinh_sach not in pols:
        raise ValueError(f"Chinh sach khong co: {chinh_sach}. Chon: {list(pols)}")
    obs, _ = env.reset(seed=seed, options={"start_day": env.val_day})
    for _ in range(max(0, min(int(ngay), env.episode_length - 1))):
        obs, *_ = env.step(pols[chinh_sach](obs, env))
    return env, obs, pols


def _ten_kho(w):
    stores = _ctx()[2]
    return stores[w] if stores and w < len(stores) else f"Kho{w}"


def _ten_sku(j):
    items = _ctx()[3]
    return items[j] if items and j < len(items) else f"SKU{j}"


def _tinh_trang_cap(env, i):
    ton = float(env.inventory[i])
    dang_ve = float(env.pipeline_orders[:, i].sum())
    cau7 = float(env.demand_history[:, i].mean()) or float(env.mean_demand[i])
    so_ngay = (ton + dang_ve) / max(cau7, 1e-6)
    return {"ton_kho": round(ton, 1), "hang_dang_ve": round(dang_ve, 1),
            "cau_tb_7_ngay": round(cau7, 2), "so_ngay_du_hang": round(so_ngay, 1),
            "fill_rate_30_ngay": round(float(env.fill_rate_per_pair[i]), 3)}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def danh_sach_kho_va_mat_hang() -> dict:
    """Liet ke 10 kho va 30 mat hang (ma M5) cung chi so dung cho cac tool khac."""
    _, bundle, stores, items = _ctx()
    W, S = bundle["demand"].shape[1:]
    return {"kho": {w: _ten_kho(w) for w in range(W)},
            "mat_hang": {j: _ten_sku(j) for j in range(S)}}


@mcp.tool()
def canh_bao_thieu_hang(ngay: int = 30, nguong_ngay: float = 2.0, kho: int | None = None) -> dict:
    """Mo phong IPPO toi `ngay` (tinh tu dau mien test) roi liet ke cac cap
    kho-mat hang co ton kho + hang dang ve du dung duoi `nguong_ngay` ngay cau."""
    env, _, _ = _run_to_day(ngay)
    ds = []
    for i in range(env.n_pairs):
        w, j = divmod(i, env.n_skus)
        if (kho is not None and w != kho) or env.demand_history[:, i].sum() == 0:
            continue                      # bo cap khong ban gi 7 ngay qua
        tt = _tinh_trang_cap(env, i)
        if tt["so_ngay_du_hang"] < nguong_ngay and tt["cau_tb_7_ngay"] > 0:
            ds.append({"kho": _ten_kho(w), "mat_hang": _ten_sku(j), "kho_id": w,
                       "sku_id": j, **tt})
    ds.sort(key=lambda r: r["so_ngay_du_hang"])
    return {"ngay_mo_phong": ngay, "nguong_ngay": nguong_ngay,
            "so_cap_canh_bao": len(ds), "chi_tiet": ds[:25]}


@mcp.tool()
def de_xuat_dat_hang(kho: int, sku: int, ngay: int = 30) -> dict:
    """De xuat luong dat hang hom nay cho mot cap (kho, sku) tai `ngay`, theo
    IPPO va 3 chinh sach co dien, kem tinh trang ton kho hien tai."""
    env, obs, pols = _run_to_day(ngay)
    i = kho * env.n_skus + sku
    de_xuat = {}
    for ten, fn in pols.items():
        a = np.asarray(fn(obs, env))
        de_xuat[ten] = {"muc_hanh_dong": int(a[i]),
                        "so_luong": float(env.action_to_qty(a)[i])}
    return {"kho": _ten_kho(kho), "mat_hang": _ten_sku(sku), "ngay": ngay,
            "tinh_trang": _tinh_trang_cap(env, i), "de_xuat": de_xuat}


@mcp.tool()
def mo_phong(chinh_sach: str = "IPPO", so_ngay: int = 90, seed: int = 1000) -> dict:
    """Chay mot chinh sach tren toan mang 300 cap trong `so_ngay` ngay (mien
    test) va tra ve KPI: tong chi phi, fill rate, so ngay-cap thieu hang."""
    cfg, bundle, *_ = _ctx()
    env = simulator.make_env(cfg, bundle, mode="test")
    pols = _policies(env)
    kq = simulator.run_episode(env, pols[chinh_sach], seed, so_ngay)
    return _kpi(kq)


@mcp.tool()
def what_if(he_so_cau: float = 1.0, so_ngay_suc_chua: float | None = None,
            lead_time_max: int | None = None, so_ngay: int = 90, seed: int = 1000) -> dict:
    """So sanh moi chinh sach trong mot kich ban gia dinh: nhan nhu cau voi
    `he_so_cau` (vd 1.5 = tang 50%), doi suc chua kho (so ngay cau) hoac thoi
    gian giao toi da."""
    cfg, bundle, *_ = _ctx()
    over = {}
    if so_ngay_suc_chua is not None:
        over["capacity_cover_days"] = float(so_ngay_suc_chua)
    if lead_time_max is not None:
        over["lead_time_max"] = int(lead_time_max)

    def factory():
        e = simulator.make_env(cfg, bundle, mode="test", env_overrides=over)
        simulator.apply_demand_shock(e, he_so_cau)
        return e

    pols = _policies(factory())
    kq = simulator.run_parallel(factory, pols, seed=seed, max_days=so_ngay)
    return {"kich_ban": {"he_so_cau": he_so_cau, **over, "so_ngay": so_ngay},
            "ket_qua": {ten: _kpi(r) for ten, r in kq.items()}}


def _kpi(kq):
    d = kq["daily"]
    return {"tong_chi_phi": round(float(d["Chi phí ngày"].sum())),
            "fill_rate": round(float(1 - d["Thiếu hàng"].sum() / max(d["Cầu"].sum(), 1e-6)), 4),
            "so_ngay_cap_thieu_hang": int((kq["stockout_matrix"] > 0).sum())}


@mcp.resource("ket-qua://tong-hop")
def ket_qua_tong_hop() -> str:
    """Ket qua danh gia chinh thuc (results/*.json) de LLM trich dan so lieu."""
    out = {}
    for name in ["iso_service.json", "summary_main_s42.json", "summary_seed42.json",
                 "regime_analysis.json"]:
        p = ROOT / "results" / name
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return json.dumps(out, ensure_ascii=False)[:20000]


def _selftest():
    print(json.dumps(de_xuat_dat_hang(0, 0, 30), ensure_ascii=False, indent=1)[:900])
    cb = canh_bao_thieu_hang(30, 2.0)
    print("canh bao:", cb["so_cap_canh_bao"], "cap;", cb["chi_tiet"][:2])
    print("mo phong:", mo_phong("IPPO", 30))
    print("what-if :", what_if(1.5, so_ngay=30)["ket_qua"])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        mcp.run()
