import json
import requests
import os
import re
import time

from client import DataCleaningEnv
from models import DataCleaningAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def create_env_with_retry(url, retries=5, delay=2):
    for i in range(retries):
        try:
            return DataCleaningEnv(base_url=url)
        except Exception as e:
            print(f"[WARN] Retry {i+1}/{retries} - Env not ready: {e}")
            time.sleep(delay)
    raise Exception("Failed to connect to environment after retries")


def call_hf_model(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an intelligent data cleaning agent. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        return response
    except Exception as e:
        print("❌ Model call failed:", e)
        return None


def run_episode():
    try:
        env_client = create_env_with_retry(ENV_URL)

        with env_client.sync() as env:
            result = env.reset()
            done = False

            print("[START]", flush=True)

            while not done:
                try:
                    obs = result.observation
                except Exception as e:
                    print("❌ Observation error:", e)
                    break

                prompt = f"""
You are a data cleaning agent.

Your PRIMARY goal is to clean ALL missing values.

STRICT RULES:
- NEVER repeat the same useless action
- ALWAYS move to a different column if one is already clean
- DO NOT fill a column if it has no missing values
- NEVER repeat the same column if already filled

DATA SAMPLE:
{obs.data_sample}

CURRENT STATE:
Columns: {obs.column_names}
Missing values: {obs.missing_values_count_per_column}
Duplicate rows: {obs.duplicate_row_count}

Return ONLY valid JSON:
{{
  "action_type": "...",
  "column_name": "...",
  "value": "..."
}}
"""

                response = call_hf_model(prompt)

                if response is None:
                    action_json = {
                        "action_type": "fill_missing",
                        "column_name": obs.column_names[0],
                        "value": "missing"
                    }
                else:
                    try:
                        action_text = response.choices[0].message.content
                    except Exception:
                        print("❌ Bad model response")
                        action_text = ""

                    if "```" in action_text:
                        action_text = action_text.split("```")[1]

                    action_text = action_text.replace("json", "").strip()
                    action_text = re.sub(r"#.*", "", action_text)
                    action_text = action_text.replace(",}", "}").replace(",]", "]")

                    try:
                        action_json = json.loads(action_text)
                        if isinstance(action_json, list):
                            action_json = action_json[0]
                    except Exception:
                        print("❌ JSON parse failed, fallback")

                        action_json = {
                            "action_type": "fill_missing",
                            "column_name": obs.column_names[0],
                            "value": "missing"
                        }

                valid_actions = [
                    "fill_missing",
                    "drop_rows_with_missing",
                    "remove_duplicates",
                    "finish_cleaning"
                ]

                action_type = action_json.get("action_type") if isinstance(action_json, dict) else None

                if action_type == "fill_missing_values":
                    print("[WARN] Fixing action_type: fill_missing_values → fill_missing")
                    action_json["action_type"] = "fill_missing"

                elif action_type not in valid_actions:
                    print(f"[WARN] Invalid action_type: {action_type}, using fallback")

                    fallback_value = "missing"
                    if isinstance(action_json, dict):
                        fallback_value = action_json.get("value") or "missing"

                    action_json = {
                        "action_type": "fill_missing",
                        "column_name": obs.column_names[0],
                        "value": str(fallback_value)
                    }

                if action_json.get("action_type") == "fill_missing":
                    col = action_json.get("column_name")
                    if obs.missing_values_count_per_column.get(col, 0) == 0:
                        for c, v in obs.missing_values_count_per_column.items():
                            if v > 0:
                                action_json["column_name"] = c
                                break

                if "value" in action_json and action_json["value"] is not None:
                    action_json["value"] = str(action_json["value"])

                try:
                    action = DataCleaningAction(**action_json)
                except Exception as e:
                    print("❌ Action creation failed:", e)
                    break

                print(f"[STEP] action={action_json}", flush=True)

                try:
                    result = env.step(action)
                except Exception as e:
                    print("❌ Step failed:", e)
                    break

                obs = result.observation

                if sum(obs.missing_values_count_per_column.values()) == 0:
                    print("[INFO] Data cleaned → stopping early")
                    break

                done = obs.done

            print("[END] success=true", flush=True)

    except Exception as e:
        print("❌ FATAL ERROR:", e)
        print("[END] success=false", flush=True)


if __name__ == "__main__":
    run_episode()