"""
env/inventory_env.py
====================
Moi truong Gymnasium cho bai toan Quan ly Ton kho Da Kho - Da SKU.

PHIEN BAN v2 - cac loi lam huan luyen khong hoi tu da duoc sua, danh dau [V2-n].

  [V2-1] SUC CHUA THEO TUNG KHO.
         Ban cu: suc_chua = trung_binh(cau ngay cua cac kho) * capacity_cover_days,
         roi ap CUNG MOT con so cho ca 10 kho. Vi CA_3 ban 217 dv/ngay con WI_1
         chi ban 58 dv/ngay, kho lon chi chua duoc 1,4 ngay cau (khong the nao
         dat fill rate 85%) trong khi kho nho chua duoc 5,2 ngay. Agent bi ep
         vao mot bai toan vo nghiem -> phat tran kho luon duong -> khong hoi tu.
         Nay suc chua tinh RIENG cho tung kho theo chinh cau cua kho do.

  [V2-2] CHUAN HOA PHAN THUONG THEO TUNG CAP.
         Phan thuong tho cua mot cap dao dong tu -0,5 (SKU ban 0,1 dv/ngay) den
         -14.000 (SKU ban 143 dv/ngay). Critic dung chung trong so phai khop
         muc tieu trai 4 bac do lon -> MSE ~1.100, gradient tu value loss lon
         gap ~290 lan gradient tu policy loss. Vi max_grad_norm=0,5 ap cho TONG
         gradient, ca cum bi thu nho 0,5/28 = 0,018 lan -> actor gan nhu khong
         hoc. Nay moi cap duoc chia cho don vi chi phi rieng cua no.

  [V2-3] BANG MUC DAT HANG KHONG CON TRUNG NHAU.
         np.round(multiplier * mean_demand * lead_time) lam 35,6% so cap chi con
         2 muc phan biet duoc trong 6 muc (vd [0,0,1,1,2,3]). Hai hanh dong khac
         nhau cho ket qua y het nhau -> gradient trai nguoc nhau, entropy khong
         giam. Nay bang muc dat hang duoc ep tang nghiem ngat.

  [V2-4] QUAN SAT CO TIN HIEU CAP KHO.
         Ham thuong phat tran kho theo TONG ton kho cua ca kho, nhung quan sat
         khong he chua thong tin do -> moi truong mat tinh Markov o dung cho
         ghep noi giua cac tac tu. Nay them 2 chieu: muc su dung suc chua cua
         kho va ty trong ton kho cua chinh cap trong kho.

  [V2-5] HANG VUOT SUC CHUA BI TU CHOI LUC NHAP, khong phai bi huy sau khi ban.
         Ban cu tinh tran kho SAU khi ban hang, nen hang da ve kho roi moi bi
         "boc hoi" - vua mat tien dat hang vua mat hang. Nay hang vuot suc chua
         bi tu choi ngay khi nhap (dung voi thuc te), van bi phat.

  [V2-6] LICH SU CAU trong quan sat khong con bi bao hoa. Chia cho
         (mean_demand * 3) roi clip 0..1 lam moi ngay ban tren 3 lan trung binh
         deu bang 1. Nay dung thang log de giu duoc thong tin duoi.

  [V2-7] info[] tra them du lieu cap kho (ton kho, suc chua, don hang) phuc vu
         mo phong truc quan trong app Streamlit.

PHIEN BAN v3:
  [V3-1] CHIA 3 MIEN: train / val / test (thay vi 2 mien train / test).
         Ban v2 dung eval_env mode="test" de chon best_model NGAY TRONG LUC
         HUAN LUYEN, tuc la viec chon checkpoint da "nhin thay" hieu nang tren
         dung mien du lieu sau nay dung de bao cao va so sanh voi baseline -
         mot dang ro ri nhe. Nay them mien "val" rieng (giua split_day va
         val_day) chi dung de chon checkpoint; mien "test" (tu val_day tro di)
         khong bao gio duoc dong tram trong luc huan luyen.

  [V3-2] CHI PHI THIEU HANG THEO GIA BAN THAT (tuy chon, co config
         use_real_price_stockout). Thay vi hang so cp_th dung chung cho ca
         300 cap, moi cap co don gia rieng:
             cp_th_pair = max(gia_ban_trung_binh(cap) * margin_ratio, cp_th_min)
         gia_ban_trung_binh lay tu M5 sell_prices.csv, uoc luong CHI TU MIEN
         TRAIN (giong cach uoc luong mean_demand). margin_ratio la gia dinh
         kinh te (M5 khong co gia von). cp_th_pair thay the cp_th trong ca
         cong thuc chi phi thieu hang lan phat muc phuc vu, va trong mau so
         chuan hoa phan thuong [V2-2] - SKU mac tien se co reward_norm_pair
         lon hon tuong ung, nen van duoc quy ve cung thang do nhu SKU re tien,
         dung co che chuan hoa per-pair da co san, khong can thiet ke lai.

  [V3-3] SUA LOI: phat SLA (phi_phat_dv) tach khoi cp_th_pair, dung hang so
         cp_th co dinh - xem docstring _calculate_reward().

  [V3-4] TIN HIEU GIAM GIA trong quan sat (leading indicator). Gia ban that
         doi theo TUAN (khuyen mai), va mot dot giam gia thuong DI TRUOC hoac
         trung luc cau tang dot bien - dung nhu hieu ung khuyen mai ban le
         kinh dien. Ban truoc chi dung gia lam THAM SO CHI PHI (tinh, [V3-2]),
         agent khong he "nhin thay" gia doi theo ngay. Nay them 1 chieu quan
         sat: muc do dang giam gia so voi gia trung binh cua chinh cap do
         (dung price_series.npy, doc tu sell_prices.csv theo tung ngay).

PHIEN BAN P0 (sua truoc khi huan luyen lai lan cuoi, xem
README_CLAUDE_CODE_KLTN.md):
  [P0-1] SUA LOI TRAN KHO: chi tu choi hang MOI VE vuot dung luong con
         trong, khong con phan bo phan vuot theo ty trong tren CA ton kho cu
         lan hang moi. Ban [V2-5] tinh tong (ton_cu + hang_moi) roi tru theo
         ty trong -> vo tinh "an" ca vao ton kho cu dang nam hop le trong suc
         chua (vd ton cu=90, suc chua=100, hang moi=20 -> ban cu tu choi
         10/110 tren ca hai nguon, lam ton kho cu giam oan). Nay tinh
         free_capacity = suc_chua - ton_cu, chi tu choi dung phan hang moi
         vuot qua free_capacity; ton kho cu khong bao gio bi buoc nay giam.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class MultiWarehouseInventoryEnv(gym.Env):
    """Moi truong Gymnasium cho Quan ly Ton kho Da Kho - Da SKU."""

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 demand_data: Optional[np.ndarray] = None,
                 calendar_features: Optional[np.ndarray] = None,
                 price_per_pair: Optional[np.ndarray] = None,
                 price_series: Optional[np.ndarray] = None,
                 mode: str = "train"):
        super().__init__()

        self.config = config or {}
        self.n_warehouses = int(self.config.get("n_warehouses", 10))
        self.n_skus       = int(self.config.get("n_skus", 50))
        self.n_pairs      = self.n_warehouses * self.n_skus
        self.mode         = mode                      # "train" | "test"

        # -- Tham so chi phi -------------------------------------------------
        self.cp_lk  = float(self.config.get("cp_lk",  1.0))    # luu kho / dv / ngay
        self.cp_th  = float(self.config.get("cp_th",  10.0))   # thieu hang / dv
        self.cp_dh  = float(self.config.get("cp_dh",  5.0))    # dat hang / lan dat
        self.pt_tk  = float(self.config.get("pt_tk",  5.0))    # phat tran kho / dv
        self.phi_dv = float(self.config.get("phi_dv", 3.0))    # he so phat muc phuc vu
        self.muc_dv = float(self.config.get("muc_dv", 0.85))   # nguong fill rate (beta)

        self._capacity_cfg       = self.config.get("warehouse_capacity", "auto")
        self.capacity_cover_days = float(self.config.get("capacity_cover_days", 8.0))

        self.tg_giao_min = int(self.config.get("lead_time_min", 1))
        self.tg_giao_max = int(self.config.get("lead_time_max", 3))
        self.tg_giao_tb  = 0.5 * (self.tg_giao_min + self.tg_giao_max)
        self.ch_ls       = int(self.config.get("lookback", 7))
        self.do_dai_tap  = int(self.config.get("episode_length", 365))
        self.cua_so_dv   = int(self.config.get("fill_rate_window", 30))
        # [V3-1] split_day = ranh gioi train/val; val_day = ranh gioi val/test.
        # Neu config khong co val_day (vd cau hinh test cu), val_day = split_day
        # -> mien "val" suy bien, hanh vi mode="test" giong het ban v2 (tuong
        # thich nguoc, khong lam hong test cu).
        self.split_day   = int(self.config.get("split_day", 1450))
        self.val_day     = int(self.config.get("val_day", self.split_day))

        self.use_relative_orders = bool(self.config.get("use_relative_orders", True))
        self.order_multipliers   = np.asarray(
            self.config.get("order_multipliers", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]),
            dtype=np.float32)
        self.order_levels_abs = np.asarray(
            self.config.get("order_levels", [0, 10, 20, 30, 40, 50]), dtype=np.float32)
        self.n_action_levels = (len(self.order_multipliers) if self.use_relative_orders
                                else len(self.order_levels_abs))

        # Alias tuong thich nguoc
        self.lead_time_min  = self.tg_giao_min
        self.lead_time_max  = self.tg_giao_max
        self.lookback       = self.ch_ls
        self.episode_length = self.do_dai_tap

        self.demand_data       = demand_data
        self.calendar_features = calendar_features
        self.calendar_dim = calendar_features.shape[1] if calendar_features is not None else 0
        self.price_per_pair    = (np.asarray(price_per_pair, dtype=np.float32).reshape(-1)
                                  if price_per_pair is not None else None)
        # [V3-4] Chuoi gia theo NGAY (khac price_per_pair - trung binh TINH mien
        # train dung cho chi phi). Reshape ve (T, n_pairs) de tra cuu theo ngay.
        self.price_series = (np.asarray(price_series, dtype=np.float32).reshape(
                                 price_series.shape[0], -1)
                             if price_series is not None else None)
        self.price_signal_dim = int(
            self.price_series is not None and self.price_per_pair is not None)

        # -- Cau trung binh tung cap, uoc luong CHI TU TAP HUAN LUYEN --------
        self.min_mean_demand = float(self.config.get("min_mean_demand", 0.05))
        if demand_data is not None:
            n_train = min(self.split_day, demand_data.shape[0])
            train_slice = demand_data[:n_train].reshape(n_train, -1)
            self.mean_demand = np.maximum(train_slice.mean(axis=0),
                                          self.min_mean_demand).astype(np.float32)
            self.std_demand = np.maximum(train_slice.std(axis=0), 1e-3).astype(np.float32)
        else:
            self.mean_demand = np.full(self.n_pairs, 10.0, dtype=np.float32)
            self.std_demand  = np.full(self.n_pairs, 3.0, dtype=np.float32)

        # [V2-1] Suc chua RIENG cho tung kho, theo cau cua chinh kho do.
        cau_tung_kho = self.mean_demand.reshape(
            self.n_warehouses, self.n_skus).sum(axis=1)                  # (n_wh,)
        if isinstance(self._capacity_cfg, str) and self._capacity_cfg.lower() == "auto":
            self.suc_chua_kho = (cau_tung_kho * self.capacity_cover_days).astype(np.float32)
        else:
            self.suc_chua_kho = np.full(self.n_warehouses, float(self._capacity_cfg),
                                        dtype=np.float32)
        self.cau_tung_kho = cau_tung_kho.astype(np.float32)

        # [V2-3] Bang muc dat hang TANG NGHIEM NGAT theo tung cap
        self.order_qty_table = self._build_order_table()
        self.order_levels = (self.order_levels_abs if not self.use_relative_orders
                             else self.order_multipliers)

        # Thang chuan hoa quan sat
        self.obs_cover_days = float(self.config.get("obs_cover_days", 20.0))
        self.inv_max = (self.mean_demand * self.obs_cover_days).astype(np.float32)

        # [V3-2] Don gia thieu hang RIENG cho tung cap, tu gia ban that (neu bat).
        # Rơi ve hang so cp_th dung chung khi khong bat, hoac khong co du lieu gia.
        self.use_real_price_stockout = bool(
            self.config.get("use_real_price_stockout", False))
        self.margin_ratio = float(self.config.get("margin_ratio", 0.3))
        self.cp_th_min    = float(self.config.get("cp_th_min", 0.5))
        if self.use_real_price_stockout and self.price_per_pair is not None:
            self.cp_th_pair = np.maximum(
                self.price_per_pair * self.margin_ratio, self.cp_th_min
            ).astype(np.float32)
        else:
            self.cp_th_pair = np.full(self.n_pairs, self.cp_th, dtype=np.float32)

        # [V2-2] Don vi chi phi rieng cua tung cap, dung chuan hoa phan thuong.
        # Chon = cp_th_pair * mean_demand + cp_dh  (chi phi cua "mot ngay xau
        # dien hinh": het hang ca ngay, cong mot lan dat hang). Hang so cp_dh
        # giu cho mau so khong tien ve 0 o cac SKU gan nhu khong ban.
        self.reward_norm_pair = (self.cp_th_pair * self.mean_demand + self.cp_dh).astype(np.float32)
        self.normalize_reward_per_pair = bool(
            self.config.get("normalize_reward_per_pair", True))
        self.reward_scale = float(self.config.get("reward_scale", 1.0))

        # -- Khong gian hanh dong & quan sat ---------------------------------
        self.action_space = spaces.MultiDiscrete([self.n_action_levels] * self.n_pairs)

        # obs = ton kho(1) + ton kho+pipeline theo ngay cau(1) + lich su cau(L)
        #     + pipeline(LT) + fill_rate(1) + dac trung quy mo(1)
        #     + [V2-4] muc su dung kho(1) + ty trong trong kho(1)
        #     + [V3-4] tin hieu giam gia (0 hoac 1)
        #     + day_of_week(7) + calendar(cal_dim)
        self.obs_per_pair = (1 + 1 + self.lookback + self.lead_time_max
                             + 1 + 1 + 1 + 1 + self.price_signal_dim
                             + 7 + self.calendar_dim)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_pairs, self.obs_per_pair), dtype=np.float32)

        self.scale_feat = np.clip(
            np.log1p(self.mean_demand) / np.log1p(150.0), 0.0, 1.0).astype(np.float32)

        self.current_step = 0
        self.inventory = None
        self.pipeline_orders = None

    # ------------------------------------------------------------------ #
    def _build_order_table(self) -> np.ndarray:
        """[V2-3] Bang luong dat hang (n_pairs, n_levels), TANG NGHIEM NGAT.

        Muc 0 luon bang 0 (khong dat hang). Cac muc con lai lam tron len va ep
        tang it nhat 1 don vi so voi muc truoc, de khong bao gio co hai hanh
        dong cho cung mot ket qua.
        """
        if not self.use_relative_orders:
            return np.tile(self.order_levels_abs,
                           (self.n_pairs, 1)).astype(np.float32)

        raw = (self.order_multipliers[None, :] *
               self.mean_demand[:, None] * self.tg_giao_tb)
        tbl = np.ceil(raw).astype(np.float32)
        tbl[:, 0] = 0.0
        # ep tang nghiem ngat tu trai sang phai
        for k in range(1, tbl.shape[1]):
            tbl[:, k] = np.maximum(tbl[:, k], tbl[:, k - 1] + 1.0)
        # muc 0 phai la 0 ngay ca sau khi ep tang
        tbl[:, 0] = 0.0
        return tbl

    def action_to_qty(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.int64)
        return self.order_qty_table[np.arange(self.n_pairs), action]

    def qty_to_action(self, target_qty: np.ndarray) -> np.ndarray:
        diffs = np.abs(self.order_qty_table -
                       np.asarray(target_qty, dtype=np.float32)[:, None])
        return np.argmin(diffs, axis=1).astype(np.int64)

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        he_so_bd = float(self.config.get("initial_cover_days", 3.0))
        self.inventory = np.round(self.mean_demand * he_so_bd).astype(np.float32)

        self.pipeline_orders    = np.zeros((self.lead_time_max, self.n_pairs), dtype=np.float32)
        self.demand_history     = np.zeros((self.lookback, self.n_pairs), dtype=np.float32)
        self.demand_fr_window   = np.zeros((self.cua_so_dv, self.n_pairs), dtype=np.float32)
        self.stockout_fr_window = np.zeros((self.cua_so_dv, self.n_pairs), dtype=np.float32)
        self.fill_rate_per_pair = np.ones(self.n_pairs, dtype=np.float32)

        if self.demand_data is not None:
            T = self.demand_data.shape[0]
            L = self.do_dai_tap
            # [V3-1] 3 mien KHONG CHONG LAN: train [0,split_day) < val
            # [split_day,val_day) < test [val_day, T). Episode luon nam
            # gon trong mien cua no (khong the "tran" sang mien ke ben).
            if self.mode == "test":
                lo = self.val_day
                hi = max(T - L, lo + 1)
            elif self.mode == "val":
                lo = self.split_day
                hi = max(self.val_day - L, lo + 1)
            else:
                lo = 0
                hi = max(self.split_day - L, 1)
            self.start_day = int(self.np_random.integers(lo, hi))
        else:
            self.start_day = 0

        self.current_step = 0
        return self._get_observation(), {}

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)

        # -- B1: Hanh dong -> luong dat hang --------------------------------
        order_qty = self.action_to_qty(action).astype(np.float32)

        # -- B2: Nhan hang tu pipeline --------------------------------------
        hang_ve = self.pipeline_orders[0].copy()

        # -- [P0-1] Tu choi hang vuot suc chua NGAY LUC NHAP, CHI TU PHAN
        # HANG MOI VE. Ban truoc ([V2-5]) tinh tong (ton kho cu + hang moi)
        # roi phan bo phan vuot theo TY TRONG tren ca hai nguon -> co the
        # "an" vao ca ton kho cu dang nam hop le trong suc chua (vd ton cu=90,
        # suc chua=100, hang moi ve=20 -> ban truoc tu choi 10/110 tren CA
        # ton cu lan hang moi, lam ton kho cu cung bi giam). Nay chi tinh
        # dung luong con trong (free_capacity = suc_chua - ton_cu), roi tu
        # choi dung phan hang moi vuot qua dung luong con trong do; ton kho
        # cu KHONG BAO GIO bi giam boi buoc nay.
        inv_before_wh = self.inventory.reshape(self.n_warehouses, self.n_skus)
        incoming_wh   = hang_ve.reshape(self.n_warehouses, self.n_skus)

        used_capacity = inv_before_wh.sum(axis=1)
        free_capacity = np.maximum(self.suc_chua_kho - used_capacity, 0.0)
        incoming_total = incoming_wh.sum(axis=1)

        accept_ratio = np.ones(self.n_warehouses, dtype=np.float32)
        vuot_kho = incoming_total > free_capacity
        accept_ratio[vuot_kho] = (free_capacity[vuot_kho]
                                  / np.maximum(incoming_total[vuot_kho], 1e-6))

        accepted_wh = incoming_wh * accept_ratio[:, None]
        rejected_wh = incoming_wh - accepted_wh

        self.inventory = (inv_before_wh + accepted_wh).reshape(-1).astype(np.float32)
        overflow = rejected_wh.reshape(-1).astype(np.float32)

        # -- B3: Dich pipeline, ghi don hang moi ----------------------------
        self.pipeline_orders = np.roll(self.pipeline_orders, shift=-1, axis=0)
        self.pipeline_orders[-1] = 0.0
        lead_times = self.np_random.integers(
            self.tg_giao_min, self.tg_giao_max + 1, size=self.n_pairs)
        np.add.at(self.pipeline_orders,
                  (lead_times - 1, np.arange(self.n_pairs)), order_qty)

        # -- B4: Cau phat sinh, ban hang ------------------------------------
        if self.demand_data is not None:
            day_idx = min(self.start_day + self.current_step,
                          self.demand_data.shape[0] - 1)
            demand = self.demand_data[day_idx].reshape(-1).astype(np.float32)
        else:
            demand = self.np_random.poisson(lam=self.mean_demand).astype(np.float32)

        sold     = np.minimum(self.inventory, demand)
        stockout = demand - sold
        self.inventory = self.inventory - sold

        # -- B5: Cap nhat lich su -------------------------------------------
        self.demand_history = np.roll(self.demand_history, shift=-1, axis=0)
        self.demand_history[-1] = demand

        self.demand_fr_window = np.roll(self.demand_fr_window, shift=-1, axis=0)
        self.demand_fr_window[-1] = demand
        self.stockout_fr_window = np.roll(self.stockout_fr_window, shift=-1, axis=0)
        self.stockout_fr_window[-1] = stockout

        tong_nc = self.demand_fr_window.sum(axis=0)
        tong_th = self.stockout_fr_window.sum(axis=0)
        self.fill_rate_per_pair = np.where(
            tong_nc > 1e-6, 1.0 - tong_th / np.maximum(tong_nc, 1e-6), 1.0
        ).astype(np.float32)

        # -- B6: Reward ------------------------------------------------------
        raw_reward, local_rewards, cost_breakdown = self._calculate_reward(
            self.inventory, stockout, order_qty, overflow, self.fill_rate_per_pair)

        # [V2-2] Chuan hoa theo tung cap -> moi tac tu co cung do lon phan thuong
        if self.normalize_reward_per_pair:
            local_scaled = local_rewards / (self.reward_norm_pair * self.reward_scale)
        else:
            local_scaled = local_rewards / (100.0 * self.reward_scale)
        scaled_reward = float(local_scaled.mean())

        self.current_step += 1
        terminated = False
        truncated  = self.current_step >= self.episode_length

        inv_wh_end = self.inventory.reshape(self.n_warehouses, self.n_skus)
        info = {
            "demand":          float(demand.sum()),
            "demand_pairs":    demand,
            "sold":            float(sold.sum()),
            "stockout":        float(stockout.sum()),
            "stockout_pairs":  stockout,
            "overflow":        float(overflow.sum()),
            "order_qty":       float(order_qty.sum()),
            "order_qty_pairs": order_qty,
            "n_orders":        int((order_qty > 0).sum()),
            "inventory":       float(self.inventory.sum()),
            "raw_reward":      raw_reward,
            "local_rewards":   local_rewards,
            "local_scaled_rewards": local_scaled.astype(np.float32),
            "fill_rate_mean":  float(self.fill_rate_per_pair.mean()),
            "cost_holding":         cost_breakdown["cost_holding"],
            "cost_stockout":        cost_breakdown["cost_stockout"],
            "cost_ordering":        cost_breakdown["cost_ordering"],
            "cost_overflow":        cost_breakdown["cost_overflow"],
            "cost_service_penalty": cost_breakdown["cost_service_penalty"],
            "cost_total":           sum(cost_breakdown.values()),
            "inventory_pairs": self.inventory.copy(),
            # [V2-7] Du lieu cap kho cho truc quan hoa
            "inv_wh":       inv_wh_end.sum(axis=1).astype(np.float32),
            "cap_wh":       self.suc_chua_kho.copy(),
            "util_wh":      (inv_wh_end.sum(axis=1) / self.suc_chua_kho).astype(np.float32),
            "overflow_wh":  overflow.reshape(self.n_warehouses, self.n_skus).sum(axis=1),
            "order_wh":     order_qty.reshape(self.n_warehouses, self.n_skus).sum(axis=1),
            "received":          float(hang_ve.sum()),           # tuong thich nguoc: hang du kien ve
            "received_accepted": float(accepted_wh.sum()),       # [P0-1] hang MOI thuc su duoc nhap
            "day_index":    int(min(self.start_day + self.current_step - 1,
                                    (self.demand_data.shape[0] - 1)
                                    if self.demand_data is not None else 0)),
        }
        return self._get_observation(), scaled_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        day_of_week = np.zeros(7, dtype=np.float32)
        if self.calendar_features is not None:
            day_idx = min(self.start_day + self.current_step,
                          len(self.calendar_features) - 1)
            cal_vec = self.calendar_features[day_idx].astype(np.float32)
            day_of_week[(self.start_day + self.current_step) % 7] = 1.0
        else:
            cal_vec = np.empty(0, dtype=np.float32)
            day_of_week[self.current_step % 7] = 1.0

        global_ctx = np.concatenate([day_of_week, cal_vec]).astype(np.float32)
        obs = np.zeros((self.n_pairs, self.obs_per_pair), dtype=np.float32)

        L  = self.lookback
        LT = self.lead_time_max
        md = self.mean_demand[:, None]

        inv_wh   = self.inventory.reshape(self.n_warehouses, self.n_skus)
        tong_kho = inv_wh.sum(axis=1, keepdims=True)
        util     = np.repeat(tong_kho / self.suc_chua_kho[:, None], self.n_skus,
                             axis=1).reshape(-1)
        share    = (inv_wh / np.maximum(tong_kho, 1e-6)).reshape(-1)

        c = 0
        # ton kho quy ve so ngay cau
        obs[:, c] = self.inventory / self.inv_max;                        c += 1
        # ton kho + pipeline (vi the ton kho) quy ve so ngay cau
        obs[:, c] = (self.inventory + self.pipeline_orders.sum(axis=0)) / self.inv_max
        c += 1
        # [V2-6] Lich su cau: thang log de khong bao hoa
        obs[:, c:c+L] = np.log1p(self.demand_history.T) / np.log1p(md * 10.0 + 1.0)
        c += L
        obs[:, c:c+LT] = self.pipeline_orders.T / self.inv_max[:, None];   c += LT
        obs[:, c] = self.fill_rate_per_pair;                               c += 1
        obs[:, c] = self.scale_feat;                                       c += 1
        # [V2-4] Tin hieu ghep noi cap kho
        obs[:, c] = util;                                                  c += 1
        obs[:, c] = share * self.n_skus / 3.0;                             c += 1
        # [V3-4] Tin hieu giam gia: 0 = gia binh thuong/tang, cang gan 1 la
        # cang giam gia sau so voi gia trung binh cua CHINH cap do.
        if self.price_signal_dim:
            day_idx = min(self.start_day + self.current_step,
                          len(self.price_series) - 1)
            gia_hom_nay = self.price_series[day_idx]
            ty_le_gia = gia_hom_nay / np.maximum(self.price_per_pair, 1e-6)
            obs[:, c] = np.clip(1.0 - ty_le_gia, 0.0, 1.0);                c += 1
        obs[:, c:] = global_ctx[None, :]

        return np.clip(obs, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def _calculate_reward(self, tk, th, sl_dh, tran, fill_rate):
        """Ham thuong phan ra cuc bo cho tung cap kho-SKU (don vi TIEN GOC).

        [V3-2] cp_th_pair thay the hang so cp_th trong CHI PHI THIEU HANG THAT
        (chi_phi_th): don gia rieng cho tung cap, tu gia ban that neu bat
        use_real_price_stockout.

        [V3-3] SUA LOI: phi_phat_dv (phat vi pham SLA 85%) KHONG con dung
        cp_th_pair, ma dung hang so cp_th CO DINH. Ly do: "phai dat 85% fill
        rate" la muc tieu quan tri co dinh, khong nen ty le theo gia SKU. Ban
        truoc dung chung cp_th_pair cho ca 2 viec -> khi bat gia that (cp_th_pair
        trung binh ~1.2, thay vi hang so cu 10), suc ep SLA yeu di ~8 lan mot
        cach khong chu y, agent hoc ra fill rate ~53% (thap hon ca 3 baseline
        co dien) van "re" hon giu du hang, mac du tong chi phi trong ve thap.
        """
        chi_phi_lk = self.cp_lk * tk
        chi_phi_th = self.cp_th_pair * th
        chi_phi_dh = self.cp_dh * (sl_dh > 0)
        phi_tran   = self.pt_tk * tran

        shortfall   = np.maximum(0.0, self.muc_dv - fill_rate)
        phi_phat_dv = self.phi_dv * self.cp_th * shortfall * self.mean_demand

        R_local = -(chi_phi_lk + chi_phi_th + chi_phi_dh + phi_tran + phi_phat_dv)

        breakdown = {
            "cost_holding":         float(np.sum(chi_phi_lk)),
            "cost_stockout":        float(np.sum(chi_phi_th)),
            "cost_ordering":        float(np.sum(chi_phi_dh)),
            "cost_overflow":        float(np.sum(phi_tran)),
            "cost_service_penalty": float(np.sum(phi_phat_dv)),
        }
        return float(np.sum(R_local)), R_local.astype(np.float32), breakdown
