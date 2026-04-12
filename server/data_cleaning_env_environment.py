# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data Cleaning Env Environment Implementation.
"""

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

    # 🔥 CRITICAL: DEFINE TASKS FOR VALIDATOR
    TASKS = [
        {
            "id": "easy-clean",
            "description": "Simple dataset with missing values",
        },
        {
            "id": "medium-clean",
            "description": "Dataset with missing + duplicates",
        },
        {
            "id": "hard-clean",
            "description": "Dataset with missing, duplicates, inconsistent values",
        },
    ]

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
    def reset(self, task_id: str = "easy-clean"):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.total_reward = 0.0

        if task_id == "easy-clean":
            data = {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, None, 30],
                "city": ["Delhi", "Mumbai", "Pune"]
            }

        elif task_id == "medium-clean":
            data = {
                "name": ["Alice", "Bob", "Alice", "Charlie"],
                "age": [25, None, 25, None],
                "city": ["Delhi", "Mumbai", "Delhi", None]
            }

        elif task_id == "hard-clean":
            data = {
                "name": ["Alice", "Bob", "Alice", "Charlie", "Bob"],
                "age": [25, None, 25, None, None],
                "city": ["Delhi", "Mumbai", "Delhi", None, "mumbai"]
            }

        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.df = pd.DataFrame(data)
        self.original_df = self.df.copy()

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

        new_missing = self.df.isnull().sum().sum()
        new_rows = len(self.df)
        new_duplicates = self.df.duplicated().sum()

        # REWARD
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

        if new_missing == 0 and new_duplicates == 0:
            reward_score += 5.0
            done = True

        if self._state.step_count >= 10:
            done = True

        # 🔥 GRADER
        score = self._compute_score(new_missing, new_duplicates)

        obs = self._get_obs()

        self.total_reward += reward_score
        obs.reward = self.total_reward
        obs.done = done

        # 🔥 REQUIRED FOR VALIDATOR
        obs.score = float(score)
        # self._state.score = float(score)

        # return StepResult(
        #     observation=obs,
        #     reward=float(self.total_reward),
        #     done=done
        # )
        return obs

    # =========================
    # GRADER
    # =========================
    def _compute_score(self, missing, duplicates):
        total_initial = self.initial_missing + self.initial_duplicates

        if total_initial <= 0:
            return 0.5

        current_total = missing + duplicates
        progress = 1 - (current_total / total_initial)

        # 🔥 SAFE RANGE BUFFER
        eps = 0.1   # buffer to avoid edge cases

        if progress <= 0:
            return 0.1
        elif progress >= 1:
            return 0.9
        else:
            return float(max(0.1, min(0.9, progress)))

    # =========================
    # OBS
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