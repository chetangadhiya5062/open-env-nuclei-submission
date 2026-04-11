# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data Cleaning Env Environment Implementation.
"""

import pandas as pd
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataCleaningAction, DataCleaningObservation
except ImportError:
    from models import DataCleaningAction, DataCleaningObservation


class DataCleaningEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df: pd.DataFrame = None
        self.original_df: pd.DataFrame = None
        self.total_reward = 0.0
        self.initial_missing = 0
        self.initial_duplicates = 0

    # =========================
    # RESET
    # =========================
    def reset(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.total_reward = 0.0

        data = {
            "name": ["Alice", "Bob", "Alice", "Charlie", "David"],
            "age": [25, None, 25, 30, None],
            "city": ["Delhi", "Mumbai", "Delhi", None, "Pune"]
        }

        self.df = pd.DataFrame(data)
        self.original_df = self.df.copy()

        # 🔥 STORE INITIAL STATE (FOR SCORING)
        self.initial_missing = self.df.isnull().sum().sum()
        self.initial_duplicates = self.df.duplicated().sum()

        return self._get_obs()

    # =========================
    # STEP
    # =========================
    def step(self, action: DataCleaningAction):
        self._state.step_count += 1

        prev_missing = self.df.isnull().sum().sum()
        prev_rows = len(self.df)
        prev_duplicates = self.df.duplicated().sum()

        reward_score = 0.0
        done = False

        # =========================
        # APPLY ACTION
        # =========================
        if action.action_type == "fill_missing":
            if action.column_name in self.df.columns:
                if self.df[action.column_name].isnull().sum() == 0:
                    reward_score -= 0.5
                else:
                    fill_value = action.value if action.value is not None else "missing"
                    self.df[action.column_name] = self.df[action.column_name].fillna(fill_value)

        elif action.action_type == "drop_rows_with_missing":
            self.df = self.df.dropna()

        elif action.action_type == "remove_duplicates":
            self.df = self.df.drop_duplicates()

        elif action.action_type == "finish_cleaning":
            done = True

        # =========================
        # NEW STATE
        # =========================
        new_missing = self.df.isnull().sum().sum()
        new_rows = len(self.df)
        new_duplicates = self.df.duplicated().sum()

        # =========================
        # REWARD LOGIC
        # =========================
        if action.action_type == "fill_missing":
            if prev_missing == new_missing:
                reward_score -= 2.0
            elif new_missing < prev_missing:
                reward_score += 2.0

        elif action.action_type == "remove_duplicates":
            if new_duplicates < prev_duplicates:
                reward_score += 2.0
            else:
                reward_score -= 0.5

        elif action.action_type == "drop_rows_with_missing":
            if new_rows < prev_rows:
                reward_score -= 2.0
            else:
                reward_score -= 1.0

        # BONUS
        if new_missing == 0 and new_duplicates == 0:
            reward_score += 5.0
            done = True

        # =========================
        # STEP LIMIT
        # =========================
        if self._state.step_count >= 10:
            done = True

        # =========================
        # COMPUTE SCORE (🔥 CRITICAL FIX)
        # =========================
        score = self._compute_score(new_missing, new_duplicates)

        # =========================
        # OBSERVATION
        # =========================
        obs = self._get_obs()

        self.total_reward += reward_score
        obs.reward = self.total_reward
        obs.done = done

        # 🔥 ADD SCORE TO STATE (REQUIRED)
        self._state.score = score

        return obs

    # =========================
    # 🔥 GRADER (MOST IMPORTANT)
    # =========================
    def _compute_score(self, missing, duplicates):
        total_initial = self.initial_missing + self.initial_duplicates

        if total_initial == 0:
            return 0.5  # safe default

        current_total = missing + duplicates

        progress = 1 - (current_total / total_initial)

        # 🔥 STRICT RANGE (0,1)
        return max(0.05, min(0.95, progress))

    # =========================
    # OBSERVATION
    # =========================
    def _get_obs(self) -> DataCleaningObservation:
        return DataCleaningObservation(
            missing_values_count_per_column=self.df.isnull().sum().to_dict(),
            duplicate_row_count=int(self.df.duplicated().sum()),
            total_row_count=len(self.df),
            column_names=list(self.df.columns),
            data_sample=self.df.head(5).fillna("missing").to_dict(orient="records"),
        )

    @property
    def state(self) -> State:
        return self._state