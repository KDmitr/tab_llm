import ast
import pandas as pd
import json
from tqdm import tqdm
import time
import ast

def has_curly_brace(text):
    return '{' in text or '}' in text

def extract_json_objects(text, col_names=None):
    """Parse LLM response into a list of dicts.

    Supports two response formats:
      1. List/sequence of JSON objects  -> [{...}, {...}, ...]
      2. List of lists (positional)     -> [[v1, v2, ...], [v1, v2, ...], ...]
         Requires col_names to convert rows into dicts; rows with wrong length
         are skipped with a warning.
    Falls back to format-2 only when format-1 yields nothing.
    """
    # --- Format 1: objects {…} ---
    json_objects = []
    stack = []
    start_index = -1

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_index = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    json_str = text[start_index:i + 1]
                    try:
                        obj = json.loads(json_str)
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass

    if json_objects:
        return json_objects

    # --- Format 2: list-of-lists [[…], […], …] ---
    if col_names is None:
        return []

    outer_start = text.find('[')
    if outer_start == -1:
        return []

    # Extract the outermost [...] block
    depth = 0
    outer_end = -1
    for i in range(outer_start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                outer_end = i + 1
                break

    if outer_end == -1:
        return []

    try:
        rows = json.loads(text[outer_start:outer_end])
    except (json.JSONDecodeError, ValueError):
        try:
            rows = ast.literal_eval(text[outer_start:outer_end])
        except Exception:
            return []

    if not isinstance(rows, list):
        return []

    result = []
    n_cols = len(col_names)
    for row in rows:
        if not isinstance(row, list):
            continue
        if len(row) != n_cols:
            print(f"[WARN] list-of-lists row has {len(row)} values, expected {n_cols} — skipped")
            continue
        result.append(dict(zip(col_names, row)))

    return result

def extract_json(input_string):
    start_pos = input_string.find('[')  
    if start_pos == -1:
        return None  

    nesting_level = 0
    for i in range(start_pos, len(input_string)):
        if input_string[i] == '[':
            nesting_level += 1
        elif input_string[i] == ']':
            nesting_level -= 1
        
        if nesting_level == 0:  
            end_pos = i + 1  
            json_string = input_string[start_pos:end_pos]
            json_string.replace('\n','')
            
            try: 
                data_ = ast.literal_eval(json_string)
                return data_
            except:
                continue
    return None  


# ---------------------------------------------------------------------------
# Built-in dataset hints
# ---------------------------------------------------------------------------

DATASET_HINTS = {
    "adult": """
Rules: capital-gain,capital-loss=0 in >90% rows. hours-per-week~40. income: 0 ~75%, 1 ~25%.
relationship=0→sex=1; relationship=5→sex=0. marital-status∈{1,2}→relationship∈{0,5}. marital-status=4→relationship∈{1,2,3,4}.
income=1 correlates: age 35-60, education≥9, occupation∈{3,9}, marital-status=2, sex=1.
""".strip(),

    "phoneme": """
Rules: 5 float features (signed harmonic amplitudes / total energy).
Ranges: Aa[-1.7,4.1] mean=0.82, Ao[-1.3,4.4] mean=1.26, Dcl[-1.8,3.2] mean=0.76, Iy[-1.6,2.8] mean=0.40, Sh[-1.3,2.7] mean=0.08.
class: 0 ~70%, 1 ~30%. class=1 → higher Aa,Ao. Features weakly correlated.
""".strip(),

    "seattle_housing": """
Constraints:
- beds: int 1-6. baths: float in 0.5 steps. size: 500-5000 sqft. lot: 2000-20000 sqft
- price: right-skewed, typical 80k-1.5M, median ~450k
- price correlates with size and zipcode. baths scales with beds (not 1:1)
- zipcode: valid King County WA codes (98001-98199)
""".strip(),

    "562_cpu_small": """
Constraints:
- MMAX >= MMIN (always). CHMAX >= CHMIN (always)
- CACH ∈ {0,4,8,16,32,64,128,256}; 0 is most common
- class: 6-1150, right-skewed (~75% below 200)
- Low MYCT → high class. High class (>300): MYCT<50, MMAX>16000, CACH>32
""".strip(),

    "magic04": """
Rules: fWidth<=fLength. fConc1<=fConc. fConc,fConc1∈[0,1]. fAlpha∈[0,90].
class: 0=gamma ~65%, 1=hadron ~35%. gamma(0): small fAlpha, high fConc, elongated. hadron(1): large fAlpha, low fConc.
""".strip(),
}


# ---------------------------------------------------------------------------
# Zero-shot categorical encoding hints (used ONLY in zero-shot mode)
# Specifies allowed numeric codes for categorical/discrete features.
# Not included in few-shot prompts where real rows already show the values.
# ---------------------------------------------------------------------------

DATASET_ZEROSHOT_HINTS = {
    "adult": """
Cat.codes(int): workclass(0=FedGov,1=LocGov,2=Priv,3=SE-i,4=SE-ni,5=StateGov,6=NoPay) edu(0=10th,1=11th,2=12th,3=1-4,4=5-6,5=7-8,6=9th,7=AscA,8=AscV,9=BA,10=PhD,11=HS,12=MA,13=PreSch,14=Prof,15=SomeCol) marital(0=Div,1=MarAF,2=MarCiv,3=MarAbs,4=Single,5=Sep,6=Wid) occ(0=Adm,1=Army,2=Craft,3=Exec,4=Farm,5=Handlers,6=MachOp,7=OthSvc,8=PrivHouse,9=Prof,10=Protect,11=Sales,12=TechSup,13=Trans) rel(0=Husb,1=NoFam,2=OthRel,3=Child,4=Unmar,5=Wife) race(0=AI,1=API,2=Black,3=Oth,4=White) sex(0=F,1=M) country(38=US ~90%) income(0=low,1=high)
""".strip(),

    "phoneme": """
Categorical feature encoding (use these integer codes exactly):
- class: 0=nasal vowel, 1=oral vowel
""".strip(),

    "seattle_housing": """
Discrete feature allowed values:
- beds: integer, typically 1–6
- baths: float in 0.5 steps only (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
- zipcode: valid Seattle WA ZIP codes only: 98101, 98102, 98103, 98104, 98105, 98106, 98107,
  98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119, 98121, 98122, 98125, 98126,
  98133, 98134, 98136, 98144, 98146, 98154, 98155, 98166, 98168, 98177, 98178, 98195, 98199
""".strip(),

    "562_cpu_small": """
Discrete feature allowed values:
- CACH: must be one of 0, 4, 8, 16, 32, 64, 128, 256 (power-of-two cache sizes; 0 is most common)
""".strip(),

    "magic04": """
Categorical feature encoding (use these integer codes exactly):
- class: 0=gamma (signal, g), 1=hadron (background, h)
""".strip(),
}


class GEN():
    def __init__(
        self,
        gen_client,
        gen_model_nm,
        real_data,
        cols,
        gen_temperature=0.5,
        batch_size=10,
        gen_ratio=1.0,
        dataset_hint=None,       # str: ключ из DATASET_HINTS ("adult", ...) или None
        custom_hint=None,        # str: произвольная подсказка, добавляется после dataset_hint
        zero_shot=False,         # bool: если True — не передаёт реальные строки в промпт
    ) -> None:
        self.gen_client = gen_client
        self.gen_model_nm = gen_model_nm
        self.real_data = real_data
        self.cols = real_data.columns
        self.num_cols = cols
        real_data.reset_index(inplace=True, drop=True)
        
        self.tobe_refined = ""
        self.zero_shot = zero_shot

        pred_true_df = pd.get_dummies(real_data[cols])
        self.pred_Xcols = list(pred_true_df.columns)
        self.res = {} 
        self.res_df = [] 
        self.sample = None 
        self.df_syn = None
        self.res_df = []
        self.model = None
        self.dict_template = {}
        self.ress = None
        
        for col in cols:
            self.dict_template[col] = []
        
        self.gen_ratio = gen_ratio
        self.batch_size = batch_size
        self.output_batch_size = max(1, round(batch_size * gen_ratio))

        temp = []
        for i in range(self.output_batch_size):
            temptemp = '{' + f'sample {i}' + '}'
            temp.append(temptemp)
        self.response_template = str(temp)

        self.gen_temperature = gen_temperature

        # --- Hint assembly ---
        hint_parts = []

        if dataset_hint is not None:
            key = dataset_hint.lower().strip()
            if key in DATASET_HINTS:
                hint_parts.append(DATASET_HINTS[key])
            else:
                available = list(DATASET_HINTS.keys())
                raise ValueError(
                    f"Unknown dataset_hint='{dataset_hint}'. "
                    f"Available built-in hints: {available}. "
                    f"Pass custom_hint for a fully custom hint."
                )

        if custom_hint is not None:
            hint_parts.append(custom_hint.strip())

        self.hint = "\n\n".join(hint_parts) if hint_parts else None

        # --- Zero-shot categorical encoding hint ---
        if dataset_hint is not None and zero_shot:
            key = dataset_hint.lower().strip()
            self.zeroshot_hint = DATASET_ZEROSHOT_HINTS.get(key, None)
        else:
            self.zeroshot_hint = None

        if self.zero_shot:
            print(
                "[INFO] zero_shot=True: real rows will NOT be passed in the prompt. "
                "Only column schema (and hint if provided) will be used."
            )


    def instruction(self, sample):
        """Few-shot промпт: реальные строки передаются в user message.
        
        Hint добавляется в user message (не в system) чтобы экономить контекст
        и держать constraints рядом с данными.
        """
        prompt_sys = "You generate synthetic tabular data. Output ONLY a JSON list of objects, no other text."

        prompt_user = f"Real data examples: {sample}\n"
        if self.hint:
            prompt_user += f"\n{self.hint}\n\n"
        prompt_user += (
            f"Generate {self.output_batch_size} new samples from the same distribution. "
            f"Vary values naturally — do not repeat the examples verbatim. "
            f"Respond with ONLY a JSON list."
        )
        print(f"prompt_user: {prompt_user}")
        return prompt_sys, prompt_user

    def instruction_zero_shot(self, n_rows):
        """Zero-shot промпт: только схема колонок, без реальных строк.
        Args:
            n_rows: сколько строк запросить у модели в этом вызове.
        """
        base_sys = "The ultimate goal is to produce accurate and convincing synthetic data."
        sys_parts = [base_sys]
        if self.zeroshot_hint:
            sys_parts.append(self.zeroshot_hint)
        prompt_sys = "\n\n".join(sys_parts)
        schema = list(self.cols)
        prompt_user = (
            f"Generate {n_rows} synthetic samples for a dataset "
            f"with the following fields: {schema}.\n"
            f"Each generated sample MUST be unique. Do NOT produce duplicate or nearly identical rows. "
            f"All rows should differ meaningfully across multiple fields.\n"
            f"The response should be formatted STRICTLY as a list in JSON format, "
            f"which is suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. "
            f"No additional text or numbers should precede the JSON data."
        )
        return prompt_sys, prompt_user

    def row2dict(self, rows):
        rows.reset_index(inplace=True, drop=True)
        res = []
        for i in range(len(rows)):
            example_data = {}
            row = rows.iloc[i, :]
            for x in self.cols:
                if x in self.num_cols:
                    example_data[x] = row[x]
                else:
                    example_data[x] = row[x]
            res.append(example_data)
        return str(res)


    def _call_llm(self, sys_info, user_info, label=""):
        """Один вызов LLM с MAX_RETRIES. Возвращает список распарсенных объектов или []."""
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                batch_start = time.time()
                resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm,
                    messages=[
                        {"role": "system", "content": sys_info},
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n=1,
                    max_tokens=5000000
                )
                batch_elapsed = time.time() - batch_start
                usage = resp_temp.usage
                print(
                    f"[{label}] time={batch_elapsed:.2f}s | "
                    f"input={usage.prompt_tokens}, output={usage.completion_tokens}, "
                    f"total={usage.total_tokens}"
                )
                content = resp_temp.choices[0].message.content
                if content is not None:
                    parsed = extract_json_objects(content, col_names=list(self.cols))
                    if parsed:
                        return parsed
                    print(f"[Attempt {attempt+1}/{MAX_RETRIES}] Parsed result is empty, retrying...")
                else:
                    print(f"[Attempt {attempt+1}/{MAX_RETRIES}] Response is None, retrying...")
            except Exception as e:
                print(f"[Attempt {attempt+1}/{MAX_RETRIES}] Error: {e}")
        return []

    def gen(self, batch_size, i=0):
        if self.zero_shot:
            self._gen_zero_shot(batch_size)
        else:
            self._gen_few_shot(batch_size, i)

    def _gen_few_shot(self, batch_size, i=0):
        """Оригинальная few-shot генерация — батчами по реальным строкам."""
        import re
        res = []
        end_idx = min(i + batch_size, len(self.real_data))
        self.sample = self.real_data.loc[i:end_idx, self.cols].copy()

        for j in tqdm(range(i, end_idx, self.batch_size)):
            sampled_rows = self.real_data.loc[j:j + self.batch_size - 1, self.cols].copy()
            sample = self.row2dict(sampled_rows)
            s_clean = re.sub(r'np\.(float64|int64)\((.*?)\)', r'\2', sample)

            try:
                result = ast.literal_eval(s_clean)
            except Exception as e:
                print(f"[SKIP] ast.literal_eval failed for batch j={j}: {e}")
                continue

            sys_info, user_info = self.instruction(result)
            parsed = self._call_llm(sys_info, user_info, label=f"few_shot | j={j}")
            if parsed:
                res.append(parsed)
            else:
                print(f"[SKIP] Batch j={j} failed after all retries.")

        self.res = res

    def _gen_zero_shot(self, target_rows):
        """Zero-shot генерация с дозапросами до достижения target_rows.

        Алгоритм:
          1. Запрашиваем min(target_rows, self.batch_size) строк.
          2. Считаем сколько реально получили.
          3. Если меньше target_rows — повторяем с remaining = target_rows - collected.
          4. Останавливаемся когда собрали >= target_rows или исчерпали MAX_ZERO_SHOT_ROUNDS.
        """
        MAX_ZERO_SHOT_ROUNDS = 50  # защита от бесконечного цикла
        collected = []
        round_num = 0

        print(f"[zero_shot] Target: {target_rows} rows. Batch size: {self.batch_size}")

        with tqdm(total=target_rows, desc="zero_shot rows") as pbar:
            while len(collected) < target_rows and round_num < MAX_ZERO_SHOT_ROUNDS:
                remaining = target_rows - len(collected)
                # Просим не больше batch_size за один вызов — модели проще
                n_request = min(remaining, len(self.real_data))
                round_num += 1

                print(f"[zero_shot | round {round_num}] collected={len(collected)}, "
                      f"remaining={remaining}, requesting={n_request}")

                sys_info, user_info = self.instruction_zero_shot(n_request)
                print(f"sys_info: {sys_info}")
                print(f"user_info: {user_info}")
                parsed = self._call_llm(
                    sys_info, user_info,
                    label=f"zero_shot | round {round_num} | requesting {n_request}"
                )

                if parsed:
                    collected.extend(parsed)
                    pbar.update(min(len(parsed), remaining))
                else:
                    print(f"[zero_shot | round {round_num}] No rows returned, retrying next round.")

        if len(collected) >= target_rows:
            print(f"[zero_shot] Done: collected {len(collected)} rows (target={target_rows}).")
        else:
            print(f"[zero_shot] WARNING: collected only {len(collected)}/{target_rows} rows "
                  f"after {MAX_ZERO_SHOT_ROUNDS} rounds.")

        # Упаковываем в тот же формат что и few-shot (list of lists)
        self.res = [collected]


    def process_response(self, resp_lst):
        res = {}
        for key, val in self.dict_template.items():
            res[key] = []
        self.json_err = 0
        self.no_group_err = 0
        self.var_key_err = 0
        self.dict_error = 0
        print(resp_lst)
        json_temp = extract_json(resp_lst)
           
        return pd.DataFrame(json_temp)
    
    
    def isValid(self, s):
        stack = []
        match = {'{': '}'}
        for i in s:
            if i in ['{']:
                stack.append(i)
            if i in ['}']:
                stack.pop()
        return stack == []
    
    
    def run(self, name=''):
        target_rows = round(len(self.real_data) * self.gen_ratio)
        mode_tag = "zero_shot" if self.zero_shot else "few_shot"
        print(f"[INFO] Mode: {mode_tag} | Real rows: {len(self.real_data)} | Ratio: {self.gen_ratio} | Target synthetic rows: {target_rows}")

        try:
            self.gen(len(self.real_data))
        except Exception as e:
            print(f"[ERROR] gen() failed unexpectedly: {e}")

        try:
            records = []
            for sublist in self.res:
                if sublist:
                    records.extend(sublist)

            if records:
                if len(records) > target_rows:
                    print(f"[INFO] Trimming {len(records)} generated rows to target {target_rows}")
                    records = records[:target_rows]
                elif len(records) < target_rows:
                    print(f"[WARNING] Generated {len(records)} rows, expected {target_rows}")

                df_temp = pd.DataFrame(records, columns=list(self.real_data.columns))
            else:
                print("[WARNING] No records generated, returning empty DataFrame.")
                df_temp = pd.DataFrame(columns=list(self.real_data.columns))

            df_comb = pd.concat([df_temp])

        except Exception as e:
            print(f"[ERROR] Failed to build DataFrame: {e}")
            df_comb = pd.DataFrame(columns=list(self.real_data.columns))

        try:
            filename = (name or 'output') + '.csv'
            df_comb.to_csv(filename, index=False)
            print(f"[OK] CSV saved: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

        self.df_syn = df_comb
        return df_comb