
from pathlib import Path
import subprocess
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import mean_absolute_error, brier_score_loss, accuracy_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Tuple, Dict, Optional


# используем регулярки для выявлении важных зависимостией прибыли от новостей 
RE_SPACES = re.compile(r'\s+')
RE_NUM = re.compile(r'(\d+[.,]?\d*)')
RE_PCT = re.compile(r'(\d+[.,]?\d*)\s?%')
RE_MONEY = re.compile(r'(₽|\$|€|руб|usd|eur|доллар|евро|млрд|млн|тыс)\b', re.IGNORECASE)

UP_LEX = [
    r'рост', r'вырос', r'повыш', r'выше прогноз', r'одобр', r'позитив', r'рекорд', r'соглас',
    r'утвердил', r'получил контракт', r'улучш', r'поддерж', r'погл'
]
DOWN_LEX = [
    r'сниж', r'ниже прогноз', r'паден', r'штраф', r'санкц', r'запрет', r'расслед', r'убыток',
    r'срыв', r'останов', r'делистинг', r'ритейл\s+слаб', r'плох'
]

INTENSIFIERS = [r'значительн', r'резко', r'крупн', r'существенн', r'сильно']

RUMOR_LEX = [
    r'может', r'возможно', r'планир', r'обсужда', r'рассматри', r'источник', r'сообщается',
    r'по данным', r'rumor', r'ожидается', r'прин' 
]

EVENT_LEX = {
    'EARN': [r'выручк', r'ebitda', r'прибыл', r'мсфо', r'рсбу', r'финрезульт', r'маржин', r'операционн', r'результат'],
    'GUIDE': [r'прогноз', r'guidance', r'ожида', r'пересмотр', r'повышает прогноз', r'снижает прогноз'],
    'MA': [r'покуп', r'приобрет', r'купит', r'продаст', r'слиян', r'объедин', r'долю', r'stake', r'ipo', r'spo', r'байбек', r'обратн\w*\s+выкуп'],
    'DIV': [r'дивиден'],
    'REG': [r'фас', r'регулятор', r'минфин', r'цб\b|банк\s+россии', r'санкц', r'штраф', r'запрет', r'лиценз', r'суд', r'иск', r'арбитраж', r'расслед'],
    'PROD': [r'контракт', r'поставк', r'запуст', r'презентовал', r'тендер', r'проект'],
    'MGMT': [r'назнач', r'увол', r'совет\s+директор'],
    'MACRO': [r'нефть', r'brent', r'рубл', r'инфляц', r'ставк', r'фрс', r'ецб', r'ввп', r'экспорт', r'импорт']
}

TIER1_SOURCES = [
    'reuters', 'bloomberg', 'financial times', 'wsj', 'wall street journal',
    'интерфакс', 'таcc', 'tacc', 'тасс', 'рбк', 'коммерсант', 'ведомости'
]


def norm_text(x: str) -> str:
    if not isinstance(x, str):
        return ''
    x = x.strip().lower()
    x = RE_SPACES.sub(' ', x)
    return x


def contains_any(patterns: List[str], text: str) -> bool:
    return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)


def count_hits(patterns: List[str], text: str) -> int:
    return sum(1 for pat in patterns if re.search(pat, text, re.IGNORECASE))


def topic_key(text: str) -> str:
    t = norm_text(text)
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'\b(и|в|на|с|по|к|от|до|за|для|как|о|об|без|при|из|у|что|это|а|но|или|же|не)\b', ' ', t)
    t = RE_SPACES.sub(' ', t).strip()
    if not t:
        return 'empty'
    return hashlib.sha1(t.encode('utf-8')).hexdigest()[:12]


@dataclass
class NewsLabel:
    direction: str
    magnitude_bucket: int
    horizon: str
    event_type: str
    factuality: float
    source_tier1: int
    rumor: int
    topic_key: str


def detect_event_types(text: str) -> Dict[str, int]:
    hits = {}
    for et, pats in EVENT_LEX.items():
        c = count_hits(pats, text)
        if c > 0:
            hits[et] = c
    if not hits:
        hits['OTHER'] = 1
    return hits


def detect_direction(text: str) -> str:
    up = count_hits(UP_LEX, text)
    dn = count_hits(DOWN_LEX, text)
    if up > dn and up > 0:
        return 'up'
    if dn > up and dn > 0:
        return 'down'
    return 'neutral'


def detect_magnitude_bucket(text: str, event_types: Dict[str, int]) -> int:
    has_pct = bool(RE_PCT.search(text))
    has_money = bool(RE_MONEY.search(text))
    strong_event = any(k in event_types for k in ['REG', 'MA', 'DIV', 'EARN'])
    has_intens = contains_any(INTENSIFIERS, text)

    if (has_pct or has_money) and has_intens:
        return 3
    if (has_pct or has_money) or strong_event:
        return 2
    if contains_any(RUMOR_LEX, text):
        return 1
    return 0


def detect_horizon(event_types: Dict[str, int]) -> str:
    if any(k in event_types for k in ['EARN', 'DIV']):
        return 'd1'
    if any(k in event_types for k in ['GUIDE', 'PROD', 'MGMT']):
        return 'w1'
    if any(k in event_types for k in ['REG', 'MA']):
        return 'm1'
    if 'MACRO' in event_types:
        return 'w1'
    return 'd1'


def detect_factuality(text: str) -> float:
    score = 0.0
    if RE_PCT.search(text) or RE_MONEY.search(text):
        score += 0.3
    if contains_any([r'опубликовал', r'сообщил', r'утверд', r'принял', r'одобрил', r'отчитался'], text):
        score += 0.3
    if contains_any(RUMOR_LEX, text):
        score -= 0.3
    return float(np.clip(score, 0.0, 1.0))


def is_tier1(publication: str) -> int:
    pub = norm_text(publication)
    return int(any(s in pub for s in TIER1_SOURCES))


def label_one(title: str, publication: str) -> NewsLabel:
    text = norm_text(f"{title or ''} {publication or ''}")
    et = detect_event_types(text)

    direction = detect_direction(text)
    magnitude_bucket = detect_magnitude_bucket(text, et)
    horizon = detect_horizon(et)
    factuality = detect_factuality(text)
    source_tier1 = is_tier1(publication or '')
    rumor = int(contains_any(RUMOR_LEX, text))
    tkey = topic_key(title or '')

    return NewsLabel(direction, magnitude_bucket, horizon,
                     max(et, key=et.get), factuality, source_tier1, rumor, tkey)


def winsorize(s: pd.Series, p1=0.01, p99=0.99) -> pd.Series:
    if s.empty:
        return s
    lo = s.quantile(p1)
    hi = s.quantile(p99)
    return s.clip(lower=lo, upper=hi)


def robust_scale(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    med = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25))
    if iqr == 0:
        return s - med
    return (s - med) / iqr


def daily_features(news_df: pd.DataFrame, lookback_days=20, taus=(1.0, 5.0), burst_window=30) -> pd.DataFrame:
    """Извлечение дневных новостных фич без тикеров"""
    df = news_df.copy()
    if 'publish_date' not in df.columns:
        raise ValueError("news_df must have 'publish_date' column")
    
    # Время → UTC день
    ts = pd.to_datetime(df['publish_date'], utc=True, errors='coerce')
    df['publish_ts'] = ts
    df['publish_day'] = df['publish_ts'].dt.normalize()

    # Дедуп в пределах суток по нормализованному заголовку
    df['title_norm'] = df['title'].map(lambda x: re.sub(r'\W+', ' ', norm_text(x)).strip())
    df.sort_values(['publish_day', 'title_norm', 'publish_ts'], inplace=True)
    df = df.drop_duplicates(subset=['publish_day', 'title_norm'], keep='first')

    # Метки на уровне новости
    labels: List[NewsLabel] = [
        label_one(title=row['title'], publication=row.get('publication', ''))
        for _, row in df.iterrows()
    ]
    labdf = pd.DataFrame([l.__dict__ for l in labels])
    df = pd.concat([df.reset_index(drop=True), labdf], axis=1)

    # Календарь
    days = pd.date_range(df['publish_day'].min().normalize(),
                         df['publish_day'].max().normalize(), freq='D', tz='UTC')
    out_rows = []

    # Предподсчёт дневных количеств для BurstZ
    cnt_by_day = df.groupby('publish_day').size().reindex(days, fill_value=0)
    med_30 = cnt_by_day.rolling(window=burst_window, min_periods=1).median()
    mad_30 = (cnt_by_day - med_30).abs().rolling(window=burst_window, min_periods=1).median()
    mad_30 = mad_30.replace(0, 1.0)

    # Предпосчитать s, g, conf, weights
    s = df['direction'].map({'up': 1, 'down': -1, 'neutral': 0}).fillna(0).astype(float).values
    g_map = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5}
    g = df['magnitude_bucket'].map(g_map).fillna(0.0).astype(float).values
    conf = (df['factuality'].clip(0, 1) * (0.5 + 0.5 * df['source_tier1'].astype(int))).values
    conf = conf * np.where(df['rumor'].astype(int).values == 1, 0.5, 1.0)

    # Счётчики по типам/горизонтам для окна
    et_onehot = pd.get_dummies(df['event_type'], prefix='type')
    hz_onehot = pd.get_dummies(df['horizon'], prefix='horz')
    
    for col in ['type_EARN', 'type_GUIDE', 'type_REG', 'type_MA', 'type_DIV']:
        if col not in et_onehot.columns:
            et_onehot[col] = 0
    for col in ['horz_d1', 'horz_w1', 'horz_m1']:
        if col not in hz_onehot.columns:
            hz_onehot[col] = 0
    et_onehot = et_onehot.reset_index(drop=True)
    hz_onehot = hz_onehot.reset_index(drop=True)

    # Для быстрого доступа к данным по окну
    df_idx_by_day = defaultdict(list)
    for i, d in enumerate(df['publish_day'].values):
        df_idx_by_day[d].append(i)

    # Основной цикл по дням
    for t in days:
        # Окно [t-K, t-1]
        left = t - pd.Timedelta(days=lookback_days)
        window_days = pd.date_range(left, t - pd.Timedelta(days=1), freq='D', tz='UTC')
        idxs = []
        for d in window_days:
            idxs.extend(df_idx_by_day.get(d, []))

        if idxs:
            ref = t
            dt_days = (ref - df.iloc[idxs]['publish_day']).dt.days.values.astype(float)
            
            # ядра
            phi_vals = {}
            vol_vals = {}
            for tau in taus:
                k = np.exp(-np.maximum(dt_days, 1.0) / float(tau))
                imp = s[idxs] * g[idxs] * conf[idxs] * k
                phi_vals[f'news_net_tau{tau:g}d'] = float(imp.sum())
                vol_vals[f'news_vol_tau{tau:g}d'] = float(np.abs(imp).sum())

            # агрегаты
            sub_et = et_onehot.iloc[idxs]
            sub_hz = hz_onehot.iloc[idxs]
            type_earn = float(sub_et.get('type_EARN', pd.Series()).sum())
            type_guide = float(sub_et.get('type_GUIDE', pd.Series()).sum())
            type_reg = float(sub_et.get('type_REG', pd.Series()).sum())
            type_ma = float(sub_et.get('type_MA', pd.Series()).sum())
            type_div = float(sub_et.get('type_DIV', pd.Series()).sum())

            total_w = max(1.0, len(idxs))
            
            row = {
                'day': t,
                'news_count_{}d'.format(lookback_days): int(len(idxs)),
                'burst_z_30d': float(((cnt_by_day.loc[t] - med_30.loc[t]) / (mad_30.loc[t] if mad_30.loc[t] != 0 else 1.0)) if t in cnt_by_day.index else 0.0),
                'horz_d1_share_{}d'.format(lookback_days): float(sub_hz.get('horz_d1', pd.Series()).sum() / total_w),
                'type_earn_{}d'.format(lookback_days): type_earn / total_w,
                'type_guide_{}d'.format(lookback_days): type_guide / total_w,
                'type_reg_{}d'.format(lookback_days): type_reg / total_w,
                'type_ma_{}d'.format(lookback_days): type_ma / total_w,
                'type_div_{}d'.format(lookback_days): type_div / total_w,
                'factuality_avg_{}d'.format(lookback_days): float(df.iloc[idxs]['factuality'].mean()),
                'source_tier1_share_{}d'.format(lookback_days): float(df.iloc[idxs]['source_tier1'].astype(int).mean()),
                'rumor_ratio_{}d'.format(lookback_days): float(df.iloc[idxs]['rumor'].astype(int).mean()),
                'chain_consistency_{}d'.format(lookback_days): 0.0,
                'topic_entropy_{}d'.format(lookback_days): 0.0,
                'max_news_ts_used': pd.to_datetime(df.iloc[idxs]['publish_ts']).max(),
            }
            
            row.update(phi_vals)
            row.update(vol_vals)
            
            # Consistency
            mid_tau = 5.0 if (f'news_net_tau5d' in phi_vals and f'news_vol_tau5d' in vol_vals) else float(list(taus)[0])
            num = abs(row.get(f'news_net_tau{mid_tau:g}d', 0.0))
            den = max(1e-9, row.get(f'news_vol_tau{mid_tau:g}d', 0.0))
            row[f'chain_consistency_{lookback_days}d'] = float(np.clip(num / den, 0.0, 1.0))

            # Энтропия тем
            tk = df.iloc[idxs]['topic_key'].values.tolist()
            if len(tk) > 0:
                c = Counter(tk)
                p = np.array([v / len(tk) for v in c.values()])
                ent = float(-(p * np.log(p + 1e-12)).sum())
            else:
                ent = 0.0
            row[f'topic_entropy_{lookback_days}d'] = ent

        else:
            # пустое окно
            row = {
                'day': t,
                'news_count_{}d'.format(lookback_days): 0,
                'burst_z_30d': 0.0,
                'horz_d1_share_{}d'.format(lookback_days): 0.0,
                'type_earn_{}d'.format(lookback_days): 0.0,
                'type_guide_{}d'.format(lookback_days): 0.0,
                'type_reg_{}d'.format(lookback_days): 0.0,
                'type_ma_{}d'.format(lookback_days): 0.0,
                'type_div_{}d'.format(lookback_days): 0.0,
                'factuality_avg_{}d'.format(lookback_days): 0.0,
                'source_tier1_share_{}d'.format(lookback_days): 0.0,
                'rumor_ratio_{}d'.format(lookback_days): 0.0,
                'chain_consistency_{}d'.format(lookback_days): 0.0,
                'topic_entropy_{}d'.format(lookback_days): 0.0,
                'max_news_ts_used': pd.NaT,
            }
            for tau in taus:
                row[f'news_net_tau{tau:g}d'] = 0.0
                row[f'news_vol_tau{tau:g}d'] = 0.0

        out_rows.append(row)

    feats = pd.DataFrame(out_rows)
    # Winsorize + robust-scale
    for col in [c for c in feats.columns if c.startswith('news_net_') or c.startswith('news_vol_') or c.startswith('burst_z')]:
        feats[col] = robust_scale(winsorize(feats[col].astype(float)))

    return feats


# ============================================================================
# КЛАССЫ РЕШЕНИЙ
# ============================================================================


class METAStockPredictor:
    """
    META-Stock подход для мульти-горизонтного прогнозирования прибыли на промежутке 1-20 дней.
    
    Ключевые особенности:
    - Предсказание прибыли (return) для горизонтов 1-20 дней
    - TimeSeriesSplit для временной валидации
    - MultiOutputRegressor для одновременного предсказания всех горизонтов
    - Комбинация технических индикаторов и новостных фич
    """
    
    def __init__(self, horizons: List[int] = None, n_splits: int = 5):
        """
        Args:
            horizons: Список горизонтов для предсказания (по умолчанию 1-20 дней)
            n_splits: Количество сплитов для TimeSeriesSplit
        """
        self.horizons = horizons or list(range(1, 21))
        self.n_splits = n_splits
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def create_multi_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание таргетов для всех горизонтов"""
        df = df.copy()
        
        # Группируем по тикерам для корректного вычисления таргетов
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy().sort_values('begin')
            
            # Создаем таргеты для каждого горизонта
            for horizon in self.horizons:
                ticker_data[f'return_{horizon}d'] = (
                    ticker_data['close'].shift(-horizon) / ticker_data['close'] - 1
                )
            
            # Обновляем основной датафрейм
            for horizon in self.horizons:
                df.loc[mask, f'return_{horizon}d'] = ticker_data[f'return_{horizon}d'].values
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Подготовка признаков для мульти-горизонтного прогнозирования"""
        df = df.copy()
        
        # Базовые технические индикаторы
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy().sort_values('begin')
            
            # Моментумы разных периодов
            for period in [1, 3, 5, 10, 20]:
                ticker_data[f'momentum_{period}d'] = ticker_data['close'].pct_change(period)
            
            # Волатильности
            for period in [5, 10, 20]:
                ticker_data[f'volatility_{period}d'] = (
                    ticker_data['close'].pct_change().rolling(period).std()
                )
            
            # Скользящие средние
            for period in [5, 10, 20, 50]:
                ticker_data[f'ma_{period}'] = ticker_data['close'].rolling(period).mean()
                ticker_data[f'price_to_ma_{period}'] = ticker_data['close'] / ticker_data[f'ma_{period}']
            
            # RSI
            delta = ticker_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ticker_data['rsi_14d'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = ticker_data['close'].rolling(bb_period).mean()
            bb_std_val = ticker_data['close'].rolling(bb_period).std()
            ticker_data['bb_upper'] = bb_middle + (bb_std_val * bb_std)
            ticker_data['bb_lower'] = bb_middle - (bb_std_val * bb_std)
            ticker_data['bb_position'] = (
                (ticker_data['close'] - ticker_data['bb_lower']) / 
                (ticker_data['bb_upper'] - ticker_data['bb_lower'])
            )
            
            # Объемные индикаторы
            ticker_data['volume_ma_20'] = ticker_data['volume'].rolling(20).mean()
            ticker_data['volume_ratio'] = ticker_data['volume'] / ticker_data['volume_ma_20']
            
            # Обновляем основной датафрейм
            feature_cols = [col for col in ticker_data.columns 
                           if col.startswith(('momentum_', 'volatility_', 'ma_', 'price_to_ma_', 
                                            'rsi_', 'bb_', 'volume_'))]
            for col in feature_cols:
                if col in ticker_data.columns:
                    df.loc[mask, col] = ticker_data[col].values
        
        # Новостные фичи (если есть)
        news_features = [col for col in df.columns if col.startswith(('news_', 'type_', 'horz_', 'burst_'))]
        
        # Собираем все признаки
        all_features = [col for col in df.columns 
                       if col.startswith(('momentum_', 'volatility_', 'ma_', 'price_to_ma_', 
                                        'rsi_', 'bb_', 'volume_')) or col in news_features]
        
        # Удаляем строки с NaN в признаках
        feature_df = df[all_features].fillna(0)
        
        return feature_df, all_features
    
    def train_with_timeseries_split(self, X: np.ndarray, y: np.ndarray) -> MultiOutputRegressor:
        """Обучение модели с временной валидацией"""
        print(f"   🔄 Обучение META-Stock модели с TimeSeriesSplit (n_splits={self.n_splits})...")
        
        # Базовый регрессор
        base_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # MultiOutputRegressor для предсказания всех горизонтов одновременно
        model = MultiOutputRegressor(base_model)
        
        # Временная валидация
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Обучение на fold
            model.fit(X_train, y_train)
            
            # Предсказание и оценка
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            cv_scores.append(rmse)
            
            print(f"      Fold {fold + 1}: RMSE = {rmse:.6f}")
        
        # Обучение на всех данных
        model.fit(X, y)
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        print(f"   ✓ CV RMSE: {mean_cv_score:.6f} ± {std_cv_score:.6f}")
        
        return model
    
    def predict_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предсказание прибыли для всех горизонтов"""
        print("\ META-Stock предсказание прибыли...")
        
        # Подготавливаем признаки
        features_df, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Стандартизация
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Создаем таргеты
        df_with_targets = self.create_multi_horizon_targets(df)
        
        # Удаляем строки с NaN в таргетах
        target_cols = [f'return_{h}d' for h in self.horizons]
        valid_mask = ~df_with_targets[target_cols].isna().any(axis=1)
        
        X_train = X_scaled[valid_mask]
        y_train = df_with_targets.loc[valid_mask, target_cols].values
        
        print(f"   📊 Обучаем на {len(X_train)} примерах с {len(feature_names)} признаками")
        print(f"   📊 Предсказываем прибыль для горизонтов: {self.horizons}")
        
        # Обучение модели
        self.model = self.train_with_timeseries_split(X_train, y_train)
        
        # Предсказание для всех данных
        predictions = self.model.predict(X_scaled)
        
       
        result_df = df.copy()
        for i, horizon in enumerate(self.horizons):
            result_df[f'pred_return_{horizon}d'] = predictions[:, i]
        
        return result_df


class BaselineSolution:
    """
    Baseline решение на основе скользящих средних и моментума

    Логика:
    1. Для каждого тикера вычисляем моментум (изменение цены за последние N дней)
    2. Предсказываем, что тренд продолжится (momentum continuation)
    3. Вероятность роста = сигмоида от моментума
    """

    def __init__(self, window_size: int = 5, use_ml: bool = True):
 

        self.window_size = window_size
        self.use_ml = use_ml
        self.models = {}
        self.scalers = {}
        self.feature_names = []

    def load_data(self, train_candles_path: str,
                  test_candles_path: str,
                  train_news_path: str = None,
                  test_news_path: str = None):
        """Загрузка данных с новой структурой (_2 файлы как тестовые)"""
        print("📊 Загрузка данных...")

        # Загружаем тренировочные данные
        self.train_df = pd.read_csv(train_candles_path)
        self.train_df['begin'] = pd.to_datetime(self.train_df['begin'])

        # Загружаем тестовые данные (файлы с _2)
        self.test_df = pd.read_csv(test_candles_path)
        self.test_df['begin'] = pd.to_datetime(self.test_df['begin'])

        # Объединяем для вычисления моментума (нужна история)
        self.full_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        self.full_df = self.full_df.sort_values(['ticker', 'begin'])

        print(f"   ✓ Train: {len(self.train_df)} строк")
        print(f"   ✓ Test: {len(self.test_df)} строк")

        # --- Интеграция дневных новостных фич (без тикера) с лагом t-1 ---
        # Если готовых CSV нет, попробуем их автоматически сгенерировать.
        # Скрипт-генератор ищем в scripts/news_features_no_ticker.py или в ~/Downloads/news_features_no_ticker.py
        try:
            project_root = Path(__file__).resolve().parents[1]
            processed_dir = project_root / 'data' / 'processed' / 'participants'
            raw_dir = project_root / 'data' / 'raw' / 'participants'
            processed_dir.mkdir(parents=True, exist_ok=True)

            news_train_path = processed_dir / 'train_news_daily_features.csv'
            news_test_path = processed_dir / 'test_news_daily_features.csv'

            # Автогенерация, если файлов нет
            def try_generate(news_in: Path, out_csv: Path):
                if out_csv.exists():
                    return
                if not news_in.exists():
                    print(f"   ℹ Не найден исходный news CSV: {news_in} — пропускаю автогенерацию")
                    return
                try:
                    print(f"    Генерация news-фич из {news_in.name}...")
                    news_df = pd.read_csv(news_in)
                    feat_df = daily_features(news_df, lookback_days=20, taus=(1.0, 5.0))
                    feat_df.to_csv(out_csv, index=False)
                    print(f"    Сгенерирован файл: {out_csv}")
                except Exception as ge:
                    print(f"    Не удалось сгенерировать {out_csv.name}: {ge}")

            try_generate(raw_dir / 'train_news.csv', news_train_path)
            try_generate(raw_dir / 'test_news.csv', news_test_path)

            news_feats_list = []
            if news_train_path.exists():
                nf_tr = pd.read_csv(news_train_path)
                news_feats_list.append(nf_tr)
            if news_test_path.exists():
                nf_te = pd.read_csv(news_test_path)
                news_feats_list.append(nf_te)

            if news_feats_list:
                news_feats = pd.concat(news_feats_list, ignore_index=True)
                # Приводим день к UTC-normalized и переименовываем
                if 'day' in news_feats.columns:
                    news_feats['news_day'] = pd.to_datetime(news_feats['day'], utc=True, errors='coerce').dt.normalize()
                    news_feats = news_feats.drop(columns=['day'])
                else:
                    # Попытка автоопределения колонки дня
                    for cand in ['publish_day', 'date', 'Day', 'DATE']:
                        if cand in news_feats.columns:
                            news_feats['news_day'] = pd.to_datetime(news_feats[cand], utc=True, errors='coerce').dt.normalize()
                            break
                # Нормализуем begin до дня и сдвигаем на -1 день для лога t-1
                self.full_df['begin_day'] = pd.to_datetime(self.full_df['begin'], utc=True, errors='coerce').dt.normalize()
                self.full_df['news_day'] = self.full_df['begin_day'] - pd.Timedelta(days=1)

                # merge по дню (левый join)
                if 'news_day' in news_feats.columns:
                    merged = self.full_df.merge(news_feats, on='news_day', how='left')
                    # Заполним пропуски по числовым news-фичам нулями
                    news_cols = [c for c in merged.columns if c.startswith('news_') or c.startswith('type_') or c.startswith('horz_') or c in (
                        'factuality_avg_20d', 'source_tier1_share_20d', 'rumor_ratio_20d', 'burst_z_30d', 'topic_entropy_20d', 'chain_consistency_20d'
                    )]
                    for col in news_cols:
                        if col in merged.columns:
                            merged[col] = merged[col].fillna(0)
                    self.full_df = merged
                    print("   ✓ Подмешаны дневные новостные фичи (лаг t-1)")
                else:
                    print("    Не удалось определить колонку дня в news-фичах — пропускаю merge")
            else:
                print("   ℹ Файлы дневных news-фич не найдены — работаю как чистый baseline")
        except Exception as e:
            print(f"    Ошибка при merge news-фич: {e}. Продолжаю без новостей")

    def compute_features(self):
        """Вычисление признаков (моментум, волатильность)"""
        print("\n Вычисление признаков...")

        df = self.full_df.copy()

        # Группируем по тикерам
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()

            # 1. Моментум = процентное изменение цены за window_size дней
            ticker_data['momentum'] = (
                ticker_data['close'].pct_change(self.window_size)
            )

            # 2. Волатильность = std доходностей за window_size дней
            ticker_data['volatility'] = (
                ticker_data['close'].pct_change().rolling(self.window_size).std()
            )

            # 3. Средняя цена за window_size дней
            ticker_data['ma'] = ticker_data['close'].rolling(self.window_size).mean()

            # 4. Расстояние от MA (нормализованное)
            ticker_data['distance_from_ma'] = (
                (ticker_data['close'] - ticker_data['ma']) / ticker_data['ma']
            )

            # Обновляем данные
            df.loc[mask, 'momentum'] = ticker_data['momentum'].values
            df.loc[mask, 'volatility'] = ticker_data['volatility'].values
            df.loc[mask, 'ma'] = ticker_data['ma'].values
            df.loc[mask, 'distance_from_ma'] = ticker_data['distance_from_ma'].values

        self.full_df = df
        print(" Базовые признаки вычислены")
        
        if self.use_ml:
            self._compute_extended_features()

    def _compute_extended_features(self):
        """Вычисление расширенных признаков для ML"""
        print("   🔧 Вычисление расширенных ML-признаков...")
        
        df = self.full_df.copy()
        
        # Группируем по тикерам для расширенных фич
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()
            
            # Дополнительные моментумы
            ticker_data['momentum_1d'] = ticker_data['close'].pct_change(1)
            ticker_data['momentum_10d'] = ticker_data['close'].pct_change(10)
            ticker_data['momentum_20d'] = ticker_data['close'].pct_change(20)
            ticker_data['momentum_60d'] = ticker_data['close'].pct_change(60)
            
            # Дополнительные волатильности
            ticker_data['volatility_1d'] = ticker_data['close'].pct_change().rolling(1).std()
            ticker_data['volatility_10d'] = ticker_data['close'].pct_change().rolling(10).std()
            ticker_data['volatility_20d'] = ticker_data['close'].pct_change().rolling(20).std()
            ticker_data['volatility_60d'] = ticker_data['close'].pct_change().rolling(60).std()
            
            # Скользящие средние разных периодов
            ticker_data['ma_10'] = ticker_data['close'].rolling(10).mean()
            ticker_data['ma_20'] = ticker_data['close'].rolling(20).mean()
            ticker_data['ma_60'] = ticker_data['close'].rolling(60).mean()
            
            # RSI (упрощенный)
            delta = ticker_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ticker_data['rsi_14d'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = ticker_data['close'].rolling(bb_period).mean()
            bb_std_val = ticker_data['close'].rolling(bb_period).std()
            ticker_data['bb_upper'] = bb_middle + (bb_std_val * bb_std)
            ticker_data['bb_lower'] = bb_middle - (bb_std_val * bb_std)
            ticker_data['bb_position'] = (ticker_data['close'] - ticker_data['bb_lower']) / (ticker_data['bb_upper'] - ticker_data['bb_lower'])
            
            # Объемные признаки
            ticker_data['volume_ma_20'] = ticker_data['volume'].rolling(20).mean()
            ticker_data['volume_ratio_20d'] = ticker_data['volume'] / ticker_data['volume_ma_20']
            
            # Позиция цены в диапазоне
            ticker_data['high_20d'] = ticker_data['high'].rolling(20).max()
            ticker_data['low_20d'] = ticker_data['low'].rolling(20).min()
            ticker_data['price_position_20d'] = (ticker_data['close'] - ticker_data['low_20d']) / (ticker_data['high_20d'] - ticker_data['low_20d'])
            
            # Пересечения MA
            ticker_data['ma_cross_5_20'] = (ticker_data['ma'] > ticker_data['ma_20']).astype(int)
            ticker_data['ma_cross_10_20'] = (ticker_data['ma_10'] > ticker_data['ma_20']).astype(int)
            
            # Обновляем данные
            for col in ['momentum_1d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
                       'volatility_1d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
                       'ma_10', 'ma_20', 'ma_60', 'rsi_14d', 'bb_position',
                       'volume_ratio_20d', 'price_position_20d', 'ma_cross_5_20', 'ma_cross_10_20']:
                if col in ticker_data.columns:
                    df.loc[mask, col] = ticker_data[col].values
        
        # Кросс-фичи (если есть новостные данные)
        if 'news_net_tau1d' in df.columns:
            df['momentum_5d_x_news_net_tau1d'] = df['momentum'] * df['news_net_tau1d'].fillna(0)
            df['volatility_20d_x_news_vol_tau5d'] = df['volatility_20d'].fillna(0) * df['news_vol_tau5d'].fillna(0)
            df['volume_ratio_20d_x_burst_z_30d'] = df['volume_ratio_20d'].fillna(1) * df['burst_z_30d'].fillna(0)
        
        self.full_df = df
        print("   ✓ Расширенные признаки вычислены")

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Подготовка признаков для ML"""
        # Базовые ценовые признаки
        price_features = [
            'momentum', 'momentum_1d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
            'volatility', 'volatility_1d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
            'distance_from_ma', 'rsi_14d', 'bb_position', 'volume_ratio_20d', 'price_position_20d',
            'ma_cross_5_20', 'ma_cross_10_20'
        ]
        
        # Новостные признаки
        news_features = [
            'news_net_tau1d', 'news_net_tau5d', 'news_vol_tau1d', 'news_vol_tau5d',
            'news_count_20d', 'burst_z_30d', 'type_earn_20d', 'type_guide_20d', 'type_reg_20d',
            'type_ma_20d', 'type_div_20d', 'factuality_avg_20d', 'source_tier1_share_20d',
            'rumor_ratio_20d', 'topic_entropy_20d', 'chain_consistency_20d'
        ]
        
        # Кросс-фичи
        cross_features = [
            'momentum_5d_x_news_net_tau1d', 'volatility_20d_x_news_vol_tau5d', 'volume_ratio_20d_x_burst_z_30d'
        ]
        
        # Собираем доступные признаки
        all_features = price_features + news_features + cross_features
        available_features = [f for f in all_features if f in df.columns]
        
        # Заполняем пропуски
        feature_df = df[available_features].copy()
        feature_df = feature_df.fillna(0)
        
        return feature_df, available_features

    def _create_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Создание таргетов для обучения"""
        targets = {}
        
        # Для каждого тикера вычисляем будущие доходности
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy().sort_values('begin')
            
            # Доходности
            ticker_data['return_1d'] = ticker_data['close'].shift(-1) / ticker_data['close'] - 1
            ticker_data['return_20d'] = ticker_data['close'].shift(-20) / ticker_data['close'] - 1
            
            # Направления
            ticker_data['direction_1d'] = (ticker_data['return_1d'] > 0).astype(int)
            ticker_data['direction_20d'] = (ticker_data['return_20d'] > 0).astype(int)
            
            # Обновляем в основном датафрейме
            df.loc[mask, 'return_1d'] = ticker_data['return_1d'].values
            df.loc[mask, 'return_20d'] = ticker_data['return_20d'].values
            df.loc[mask, 'direction_1d'] = ticker_data['direction_1d'].values
            df.loc[mask, 'direction_20d'] = ticker_data['direction_20d'].values
        
        return {
            'return_1d': df['return_1d'],
            'return_20d': df['return_20d'],
            'direction_1d': df['direction_1d'],
            'direction_20d': df['direction_20d']
        }

    def train_ml_models(self):
        """Обучение CatBoost моделей"""
        if not self.use_ml:
            return
            
        print("\n🤖 Обучение CatBoost моделей...")
        
        # Подготавливаем данные только для train
        train_data = self.full_df[
            self.full_df['begin'].isin(self.train_df['begin'])
        ].copy()
        
        # Создаем таргеты
        targets = self._create_targets(train_data)
        
        # Подготавливаем признаки
        features_df, feature_names = self._prepare_features(train_data)
        self.feature_names = feature_names
        
        # Удаляем строки с NaN в таргетах
        valid_mask = ~(targets['return_1d'].isna() | targets['return_20d'].isna())
        features_df = features_df[valid_mask]
        targets = {k: v[valid_mask] for k, v in targets.items()}
        
        print(f"   📊 Обучаем на {len(features_df)} примерах с {len(feature_names)} признаками")
        
        # Параметры CatBoost
        catboost_params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
        
        # Обучение моделей для доходностей
        for target_name in ['return_1d', 'return_20d']:
            print(f"   🔄 Обучение модели для {target_name}...")
            
            model = cb.CatBoostRegressor(**catboost_params)
            model.fit(features_df, targets[target_name])
            
            # Предсказания на train для оценки
            train_pred = model.predict(features_df)
            mae = mean_absolute_error(targets[target_name], train_pred)
            print(f"      MAE на train: {mae:.6f}")
            
            self.models[target_name] = model
        
        # Обучение моделей для классификации
        for target_name in ['direction_1d', 'direction_20d']:
            print(f"   🔄 Обучение модели для {target_name}...")
            
            model = cb.CatBoostClassifier(**catboost_params)
            model.fit(features_df, targets[target_name])
            
            # Калибровка вероятностей
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(features_df, targets[target_name])
            
            # Предсказания на train для оценки
            train_pred_proba = calibrated_model.predict_proba(features_df)[:, 1]
            train_pred_binary = calibrated_model.predict(features_df)
            
            brier = brier_score_loss(targets[target_name], train_pred_proba)
            accuracy = accuracy_score(targets[target_name], train_pred_binary)
            
            print(f"      Brier на train: {brier:.6f}, Accuracy: {accuracy:.4f}")
            
            self.models[target_name] = calibrated_model
        
        print("   ✓ Модели обучены и сохранены")

    def predict(self):
        """
        Создание предсказаний
        
        Если use_ml=True: использует обученные CatBoost модели
        Иначе: использует эвристику baseline
        """
        print("\n🎯 Создание предсказаний...")

        # Фильтруем только test данные
        test_data = self.full_df[
            self.full_df['begin'].isin(self.test_df['begin'])
        ].copy()

        if self.use_ml and self.models:
            # ML предсказания
            print("   🤖 Используем обученные CatBoost модели...")
            
            # Подготавливаем признаки
            features_df, _ = self._prepare_features(test_data)
            
            # Предсказания доходностей
            test_data['pred_return_1d'] = self.models['return_1d'].predict(features_df)
            test_data['pred_return_20d'] = self.models['return_20d'].predict(features_df)
            
            # Предсказания вероятностей
            test_data['pred_prob_up_1d'] = self.models['direction_1d'].predict_proba(features_df)[:, 1]
            test_data['pred_prob_up_20d'] = self.models['direction_20d'].predict_proba(features_df)[:, 1]
            
            print("   ✓ ML предсказания готовы")
            
        else:
            # Baseline эвристика
            print("   📊 Используем baseline эвристику...")
            
            # Заполняем NaN нулями (для первых строк где нет истории)
            test_data['momentum'] = test_data['momentum'].fillna(0)
            test_data['volatility'] = test_data['volatility'].fillna(0.01)
            test_data['distance_from_ma'] = test_data['distance_from_ma'].fillna(0)

            # Новостной сигнал (если фичи присутствуют после merge)
            news_signal = 0.0
            if 'news_net_tau1d' in test_data.columns:
                news_signal = news_signal + test_data['news_net_tau1d'].fillna(0.0)
            if 'news_net_tau5d' in test_data.columns:
                news_signal = news_signal + 0.5 * test_data['news_net_tau5d'].fillna(0.0)

            # Отладочная диагностика влияния новостей
            try:
                if not isinstance(news_signal, (int, float)):
                    nz_share = float((news_signal != 0).mean())
                    print(f"   🔍 News signal coverage (доля ненулевых): {nz_share:.3f}")
                    desc = news_signal.abs().describe(percentiles=[0.5, 0.75, 0.9]).to_dict()
                    print("   🔍 |news_signal| stats:", {k: (float(v) if pd.notna(v) else None) for k, v in desc.items()})

                    # Топ-тикеры по доле ненулевых и по среднему |сигнала|
                    tmp = test_data[['ticker']].copy()
                    tmp['news_signal'] = news_signal.values
                    by_ticker = tmp.groupby('ticker')
                    top_cov = by_ticker['news_signal'].apply(lambda s: float((s != 0).mean())).sort_values(ascending=False).head(10)
                    top_mag = by_ticker['news_signal'].apply(lambda s: float(s.abs().mean())).sort_values(ascending=False).head(10)
                    print("   🔝 Тикеры по coverage:")
                    for t, v in top_cov.items():
                        print(f"      {t}: {v:.3f}")
                    print("   🔝 Тикеры по среднему |news_signal|:")
                    for t, v in top_mag.items():
                        print(f"      {t}: {v:.4f}")
            except Exception as _diag_err:
                print(f"   ⚠️ Диагностика news_signal пропущена: {_diag_err}")

            # Предсказание доходности с модификацией от новостей (консервативно)
            # Для 1 дня: momentum * (0.3 + 0.1 * clip(news_signal))
            # Для 20 дней: momentum * (1.0 + 0.2 * clip(news_signal))
            if isinstance(news_signal, (int, float)):
                ns_1d = 0.0
                ns_20d = 0.0
            else:
                ns_1d = news_signal.clip(-1.0, 1.0) * 0.1
                ns_20d = news_signal.clip(-1.0, 1.0) * 0.2

            test_data['pred_return_1d'] = test_data['momentum'] * (0.3 + (ns_1d if not isinstance(ns_1d, float) else 0.0))
            test_data['pred_return_20d'] = test_data['momentum'] * (1.0 + (ns_20d if not isinstance(ns_20d, float) else 0.0))

            # Предсказание вероятности роста
            # Используем сигмоиду для преобразования моментума в вероятность
            def sigmoid(x, sensitivity=10):
                return 1 / (1 + np.exp(-sensitivity * x))

            # Влияние новостного сигнала на вероятность роста
            if isinstance(news_signal, (int, float)):
                test_data['pred_prob_up_1d'] = sigmoid(test_data['momentum'], sensitivity=10)
                test_data['pred_prob_up_20d'] = sigmoid(test_data['momentum'], sensitivity=5)
            else:
                test_data['pred_prob_up_1d'] = sigmoid(test_data['momentum'] + 0.2 * news_signal, sensitivity=10)
                test_data['pred_prob_up_20d'] = sigmoid(test_data['momentum'] + 0.1 * news_signal, sensitivity=5)

        # Clipping: вероятности в диапазоне [0.1, 0.9] для стабильности
        test_data['pred_prob_up_1d'] = test_data['pred_prob_up_1d'].clip(0.1, 0.9)
        test_data['pred_prob_up_20d'] = test_data['pred_prob_up_20d'].clip(0.1, 0.9)

        # Clipping: доходности в разумном диапазоне [-0.2, 0.2]
        test_data['pred_return_1d'] = test_data['pred_return_1d'].clip(-0.2, 0.2)
        test_data['pred_return_20d'] = test_data['pred_return_20d'].clip(-0.5, 0.5)

        self.predictions = test_data

        print(f"   ✓ Создано {len(self.predictions)} предсказаний")
        print(f"\n    Статистика предсказаний:")
        print(f"      Средняя pred_return_1d:  {test_data['pred_return_1d'].mean():.6f}")
        print(f"      Средняя pred_return_20d: {test_data['pred_return_20d'].mean():.6f}")
        print(f"      Средняя pred_prob_up_1d: {test_data['pred_prob_up_1d'].mean():.4f}")
        print(f"      Средняя pred_prob_up_20d: {test_data['pred_prob_up_20d'].mean():.4f}")

    def save_submission(self, output_path: str = "submission.csv"):
        """Сохранение submission файла"""
        print(f"\n Сохранение submission...")

        submission = self.predictions[[
            'ticker', 'begin',
            'pred_return_1d', 'pred_return_20d',
            'pred_prob_up_1d', 'pred_prob_up_20d'
        ]].copy()

        submission.to_csv(output_path, index=False)

        print(f"   ✓ Submission сохранен: {output_path}")
        print(f"   Строк: {len(submission)}")
        print(f"\n   📋 Первые строки:")
        print(submission.head(10).to_string(index=False))

    def run(self, train_candles_path: str, test_candles_path: str,
            train_news_path: str = None, test_news_path: str = None,
            output_path: str = "submission.csv", use_meta_stock: bool = True):
        """Полный пайплайн baseline решения с поддержкой META-Stock"""
        print("=" * 70)
        print("🚀 BASELINE РЕШЕНИЕ + META-STOCK")
        print("=" * 70 + "\n")

        # 1. Загрузка данных
        self.load_data(train_candles_path, test_candles_path, train_news_path, test_news_path)

        if use_meta_stock:
            # Используем META-Stock подход для мульти-горизонтного прогнозирования
            print("\n🤖 Используем META-Stock подход...")
            meta_predictor = METAStockPredictor(horizons=list(range(1, 21)), n_splits=5)
            
            # Предсказание прибыли для всех горизонтов
            predictions_df = meta_predictor.predict_returns(self.full_df)
            
            # Фильтруем только тестовые данные для submission
            test_mask = self.full_df['begin'].isin(self.test_df['begin'])
            test_predictions = predictions_df[test_mask].copy()
            
            # Создаем submission в нужном формате
            submission_cols = ['ticker', 'begin']
            for horizon in [1, 20]:  # Основные горизонты для submission
                submission_cols.extend([
                    f'pred_return_{horizon}d',
                    f'pred_prob_up_{horizon}d'
                ])
            
            # Добавляем вероятности роста (на основе предсказанной прибыли)
            for horizon in [1, 20]:
                # Вероятность роста = сигмоида от предсказанной прибыли
                returns = test_predictions[f'pred_return_{horizon}d']
                test_predictions[f'pred_prob_up_{horizon}d'] = 1 / (1 + np.exp(-10 * returns))
            
            submission = test_predictions[submission_cols].copy()
            
        else:
            # Старый подход
            # 2. Вычисление признаков
            self.compute_features()

            # 3. Обучение ML моделей (если включено)
            if self.use_ml:
                self.train_ml_models()

            # 4. Предсказание
            self.predict()

            # 5. Создание submission
            submission = self.predictions[[
                'ticker', 'begin',
                'pred_return_1d', 'pred_return_20d',
                'pred_prob_up_1d', 'pred_prob_up_20d'
            ]].copy()

        # Сохранение
        submission.to_csv(output_path, index=False)
        
        print(f"\n💾 Submission сохранен: {output_path}")
        print(f"   Строк: {len(submission)}")
        print(f"\n   📋 Первые строки:")
        print(submission.head(10).to_string(index=False))

        print("\n" + "=" * 70)
        print("✅ BASELINE + META-STOCK ГОТОВ!")
        print("=" * 70)
        print(f"\n💡 Особенности решения:")
        print(f"   • META-Stock подход для мульти-горизонтного прогнозирования")
        print(f"   • TimeSeriesSplit для временной валидации")
        print(f"   • Предсказание прибыли на горизонтах 1-20 дней")
        print(f"   • Комбинация технических индикаторов и новостных фич")
        print(f"   • MultiOutputRegressor для одновременного предсказания всех горизонтов")


class AdvancedSolution:
    """
    Продвинутое решение, объединяющее технические индикаторы, новостные фичи и градиентный бустинг.
    
    Ключевые особенности:
    - Технические индикаторы: SMA, EMA, ROC
    - Новостные фичи: TF-IDF + сентимент (с fallback на news_features_no_ticker)
    - Временная валидация с TimeSeriesSplit
    - Градиентный бустинг с гиперпараметрической оптимизацией
    - Поддержка различных форматов данных
    """
    
    def __init__(self, test_size: float = 0.2, n_splits: int = 3):
        """
        Args:
            test_size: Доля данных для hold-out теста
            n_splits: Количество сплитов для временной валидации
        """
        self.test_size = test_size
        self.n_splits = n_splits
        self.model = None
        self.scaler = None
        self.feature_names = []
        
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление технических индикаторов для каждого инструмента"""
        df = df.sort_values('begin').copy()
        
        # Группируем по тикеру
        group_cols = ['ticker'] if 'ticker' in df.columns else None
        
        def _indicators(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group['sma_5'] = group['close'].rolling(window=5).mean()
            group['sma_10'] = group['close'].rolling(window=10).mean()
            group['ema_5'] = group['close'].ewm(span=5, adjust=False).mean()
            group['ema_10'] = group['close'].ewm(span=10, adjust=False).mean()
            group['roc_1'] = group['close'].pct_change(periods=1)
            group['roc_5'] = group['close'].pct_change(periods=5)
            return group
        
        if group_cols is not None:
            df = df.groupby(group_cols).apply(_indicators).reset_index(drop=True)
        else:
            df = _indicators(df)
        
        # Заполняем NaN значения
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def extract_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение новостных фич с использованием интегрированной функции daily_features"""
        news_df = news_df.copy()
        news_df['publish_date'] = pd.to_datetime(news_df['publish_date'])
        
        # Используем интегрированную функцию daily_features
        try:
            feat_df = daily_features(news_df, lookback_days=20, taus=(1.0, 5.0))
            if 'day' in feat_df.columns:
                feat_df['dt'] = pd.to_datetime(feat_df['day'])
            return feat_df
        except Exception as e:
            print(f"   ⚠️ Ошибка при извлечении новостных фич: {e}")
            # Fallback: базовые TF-IDF фичи
            return self._extract_basic_news_features(news_df)
    
    def _extract_basic_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: базовые TF-IDF фичи"""
        text_data = news_df['title'].fillna('')
        if 'publication' in news_df.columns:
            text_data = text_data + ' ' + news_df['publication'].fillna('')
        
        # Векторизация текста
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_text = vectorizer.fit_transform(text_data)
        
        # Простой сентимент скор
        positive_words = set(['рост', 'повышение', 'прибыль', 'успех', 'прорыв', 'up', 'rise', 'gain'])
        negative_words = set(['падение', 'убыток', 'кризис', 'снижение', 'рецессия', 'down', 'fall', 'loss'])
        
        def sentiment_score(text: str) -> int:
            tokens = text.lower().split()
            pos_count = sum(1 for tok in tokens if tok in positive_words)
            neg_count = sum(1 for tok in tokens if tok in negative_words)
            return pos_count - neg_count
        
        news_df['sentiment'] = text_data.apply(sentiment_score)
        
        # Агрегация по дням
        news_df['date_only'] = news_df['publish_date'].dt.date
        aggregated_features = []
        feature_names = [f'tfidf_{i}' for i in range(X_text.shape[1])]
        
        for date, idx in news_df.groupby('date_only').groups.items():
            tfidf_vec = X_text[idx].mean(axis=0)
            sentiment = news_df.loc[idx, 'sentiment'].sum()
            
            row = {'dt': pd.Timestamp(date), 'sentiment_sum': sentiment}
            row.update({name: tfidf_vec[0, j] for j, name in enumerate(feature_names)})
            aggregated_features.append(row)
        
        feat_df = pd.DataFrame(aggregated_features)
        return feat_df
    
    def build_dataset(self, candles_path: str, news_path: str) -> pd.DataFrame:
        """Загрузка и объединение данных свечей и новостей"""
        # Загрузка данных
        candles = pd.read_csv(candles_path)
        news = pd.read_csv(news_path)
        
        # Нормализация колонок дат
        for col in ['begin', 'dt', 'date', 'timestamp']:
            if col in candles.columns:
                candles['dt'] = pd.to_datetime(candles[col])
                break
        
        for col in ['publish_date', 'dt', 'date', 'timestamp']:
            if col in news.columns:
                news['publish_date'] = pd.to_datetime(news[col])
                break
        
        # Вычисление технических индикаторов
        candles = self.compute_technical_indicators(candles)
        
        # Создание таргета: цена вырастет в следующий период?
        candles = candles.sort_values('dt')
        candles['target_up'] = (candles['close'].shift(-1) > candles['close']).astype(int)
        
        # Удаляем строки без таргета
        candles = candles.dropna(subset=['target_up'])
        
        # Извлечение новостных фич
        news_feats = self.extract_news_features(news)
        
        # Объединение по дате
        candles['date_only'] = candles['dt'].dt.date
        news_feats['date_only'] = news_feats['dt'].dt.date
        
        merged = candles.merge(news_feats.drop(columns='dt'), on='date_only', how='left')
        merged = merged.sort_values('dt')
        
        # Forward fill для пропущенных новостных фич
        news_feature_cols = [c for c in merged.columns if c.startswith('tfidf_') or c.startswith('sentiment') or c.startswith('news_')]
        merged[news_feature_cols] = merged[news_feature_cols].fillna(method='ffill').fillna(0)
        
        # Удаляем промежуточные колонки
        merged = merged.drop(columns=['date_only'])
        return merged
    
    def train_model(self, dataset: pd.DataFrame) -> tuple:
        """Обучение градиентного бустинга с временной валидацией"""
        # Разделение на признаки и таргет
        X = dataset.drop(columns=['target_up', 'dt', 'begin'])
        y = dataset['target_up']
        
        # Выбор числовых колонок
        num_cols = X.select_dtypes(include=[np.number]).columns
        X = X[num_cols]
        self.feature_names = list(X.columns)
        
        # Стандартизация признаков
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на train/hold-out
        split_idx = int(len(X_scaled) * (1 - self.test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Временная валидация для выбора гиперпараметров
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        best_model = None
        best_score = -np.inf
        
        # Сетка гиперпараметров
        learning_rates = [0.05, 0.1]
        n_estimators_list = [50, 100]
        max_depths = [3, 5]
        
        print(f"    Оптимизация гиперпараметров на {len(X_train)} примерах...")
        
        for lr in learning_rates:
            for n_estimators in n_estimators_list:
                for depth in max_depths:
                    model = GradientBoostingClassifier(
                        learning_rate=lr,
                        n_estimators=n_estimators,
                        max_depth=depth,
                        random_state=42,
                    )
                    
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        model.fit(X_tr, y_tr)
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        acc = accuracy_score(y_val, y_pred)
                        cv_scores.append(acc)
                    
                    mean_score = np.mean(cv_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = model
        
        # Обучение лучшей модели на всем train наборе
        assert best_model is not None
        best_model.fit(X_train, y_train)
        
        # Предсказания на hold-out наборе
        y_pred_test = best_model.predict_proba(X_test)[:, 1]
        
        print(f"   ✓ Лучший CV score: {best_score:.4f}")
        print(f"   ✓ Hold-out accuracy: {accuracy_score(y_test, (y_pred_test > 0.5).astype(int)):.4f}")
        
        self.model = best_model
        return best_model, y_test, pd.Series(y_pred_test, index=y_test.index)
    
    def run(self, train_candles_path: str, test_candles_path: str,
            train_news_path: str = None, test_news_path: str = None,
            output_path: str = "advanced_submission.csv"):
        """Полный пайплайн продвинутого решения с новой структурой данных"""
        print("=" * 70)
        print("🚀 ADVANCED SOLUTION (Gradient Boosting + News)")
        print("=" * 70 + "\n")
        
        # Объединяем train и test данные для обучения
        print("📊 Построение датасета...")
        train_dataset = self.build_dataset(train_candles_path, train_news_path)
        test_dataset = self.build_dataset(test_candles_path, test_news_path)
        
        # Объединяем для обучения (но сохраняем разделение)
        full_dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
        
        print(f"   ✓ Train: {len(train_dataset)} строк")
        print(f"   ✓ Test: {len(test_dataset)} строк")
        print(f"   ✓ Features: {len([c for c in full_dataset.columns if c not in ['target_up', 'dt', 'begin']])}")
        
        # Обучение модели
        print("\n🤖 Обучение модели...")
        model, y_true, y_pred = self.train_model(full_dataset)
        
        # Создание submission файла в нужном формате
        print(f"\n💾 Сохранение submission...")
        
        # Берем только test данные для submission
        test_mask = full_dataset['dt'].isin(test_dataset['dt'])
        test_subset = full_dataset[test_mask].copy()
        
        # Предсказания для test данных
        X_test = test_subset[self.feature_names]
        X_test_scaled = self.scaler.transform(X_test)
        
        test_subset['pred_return_1d'] = model.predict_proba(X_test_scaled)[:, 1] * 0.1  # Масштабируем
        test_subset['pred_return_20d'] = model.predict_proba(X_test_scaled)[:, 1] * 0.2
        test_subset['pred_prob_up_1d'] = model.predict_proba(X_test_scaled)[:, 1]
        test_subset['pred_prob_up_20d'] = model.predict_proba(X_test_scaled)[:, 1]
        
        # Клиппинг
        test_subset['pred_return_1d'] = test_subset['pred_return_1d'].clip(-0.2, 0.2)
        test_subset['pred_return_20d'] = test_subset['pred_return_20d'].clip(-0.5, 0.5)
        test_subset['pred_prob_up_1d'] = test_subset['pred_prob_up_1d'].clip(0.1, 0.9)
        test_subset['pred_prob_up_20d'] = test_subset['pred_prob_up_20d'].clip(0.1, 0.9)
        
        # Сохранение в нужном формате
        submission = test_subset[[
            'ticker', 'begin',
            'pred_return_1d', 'pred_return_20d',
            'pred_prob_up_1d', 'pred_prob_up_20d'
        ]].copy()
        
        submission.to_csv(output_path, index=False)
        
        print(f"   ✓ Submission сохранен: {output_path}")
        print(f"   Строк: {len(submission)}")
        
        print("\n" + "=" * 70)
        print("✅ ADVANCED SOLUTION ГОТОВ!")
        print("=" * 70)


def run_baseline_solution():
    """Запуск BaselineSolution с META-Stock подходом"""
    baseline = BaselineSolution(window_size=5, use_ml=True)
    baseline.run(
        train_candles_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv",
        test_candles_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv",
        train_news_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news.csv",
        test_news_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news_2.csv",
        output_path="meta_stock_submission.csv",
        use_meta_stock=True
    )


def run_advanced_solution():
    """Запуск AdvancedSolution с градиентным бустингом"""
    solution = AdvancedSolution(test_size=0.2, n_splits=3)
    solution.run(
        train_candles_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles.csv",
        test_candles_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/candles_2.csv",
        train_news_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news.csv",
        test_news_path="/Users/safroelena/Desktop/фин-хак/finam-x-hse-trade-ai-hack-forecast/data/raw/participants/news_2.csv",
        output_path="advanced_submission.csv"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        print("🚀 Запуск AdvancedSolution...")
        run_advanced_solution()
    else:
        print("🚀 Запуск BaselineSolution...")
        run_baseline_solution()

