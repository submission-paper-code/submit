# from openai import OpenAI
# client = OpenAI()
import re
import difflib
import json
import ast

import os
from openai import OpenAI

import time
import logging
from typing import List, Optional, Dict, Tuple
import tiktoken

logger = logging.getLogger(__name__)

class LLMGuidancePolicy:
    def __init__(
        self,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self._call_seq = 0

        self.cum_prompt_chars = 0
        self.cum_answer_chars = 0
        self.cum_prompt_tokens = 0
        self.cum_answer_tokens = 0

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

        self.ns_client = OpenAI(
            api_key=os.environ.get("NSCALE_SERVICE_TOKEN"),
            base_url="https://inference.api.nscale.com/v1")
        self._openai_client = OpenAI()
        self.nebius_client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                            api_key=os.environ.get("NEBIUS_API_KEY"))

    def _lengths(self, text: str) -> Tuple[int, int]:
        char_len = len(text)
        if self._enc is None:
            tok_len = 0
        else:
            tok_len = len(self._enc.encode(text))
        return char_len, tok_len

    def _is_openai_model(self, model_name: str) -> bool:
        return (model_name or "").startswith("gpt-")

    def _is_google_model(self, model_name: str) -> bool:
        return (model_name or "").startswith("google")

    def pick_mutators(
        self,
        mod,
        available_mutators: List[str],
        llm_bucket: List[str],
        historical_perf: Optional[str] = None,
        available_mutator_probs: Optional[Dict[str, float]] = None,
        current_model: Optional[str] = None,
        model_performance: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        if not current_model:
            raise ValueError(
                "LLMGuidancePolicy.pick_mutators called without a current_model "
                "(this should be handled in the caller)."
            )
        self._call_seq += 1
        call_id = self._call_seq
        logger.warning("[LLM %d] pick_mutators(): model=%s", call_id, current_model)

        system_prompt, user_prompt = self._build_prompt(
            available_mutators=available_mutators,
            historical_perf=historical_perf,
            mutator_probs=available_mutator_probs,
            current_model=current_model,
            llm_bucket=llm_bucket,
            model_performance=model_performance,
        )

        full_prompt = system_prompt + user_prompt
        p_chars, p_tokens = self._lengths(full_prompt)
        if self.verbose:
            logger.warning("\n Full Prompt \n%s", full_prompt)
            
        logger.warning("\n==== Current Prompt stats (chars=%d, tokens≈%d) ====\n",
                           p_chars, p_tokens)

        self.cum_prompt_chars += p_chars
        self.cum_prompt_tokens += p_tokens

        logger.warning(
                "\n==== Total cumulative Prompt stats: %d chars (%d tokens) ====\n",
                self.cum_prompt_chars,  self.cum_prompt_tokens)

        try:
            # logger.warning("You're here inside LLMGuidancePolicy at line 69")
            t0 = time.perf_counter()
            if self._is_openai_model(current_model):
                response = self._openai_client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            elif self._is_google_model(current_model):
                response = self.nebius_client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            else:
                response = self.ns_client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            api_latency = time.perf_counter() - t0

            content = response.choices[0].message.content
            logger.warning("[LLM %d] API latency: %.3f sec", call_id, api_latency)

            a_chars, a_tokens = self._lengths(content)

            if self.verbose:
                logger.warning("\n LLM raw response:\n%s", content)

            logger.warning("\n==== Current LLM Answer stats (chars=%d, tokens≈%d) ====\n",
                               a_chars, a_tokens)
                
            self.cum_answer_chars += a_chars
            self.cum_answer_tokens += a_tokens
            logger.warning("\n==== Total cumulative LLM Answer stats (chars=%d, tokens≈%d) ====\n",
                               self.cum_answer_chars, self.cum_answer_tokens)

            mutator_list, next_model = self._extract_mutators_list_and_model(
                model_text=content, 
                valid_mutators=available_mutators,
                valid_models=llm_bucket,)
            logger.warning(
                "[LLM %d] parsed output: mutators=%s next_model=%s",
                call_id, mutator_list, next_model
            )
            if next_model is None:
                logger.warning(
                    "LLM did not return a valid next model name (caller will choose fallback)."
                )
                # next_model = current_model
            if not mutator_list:
                if self.verbose:
                    logger.warning("LLM did not return a valid list of mutators.")
                return None, next_model
            return mutator_list, next_model

        except Exception as e:
            logger.warning("ChatCompletion failed: %s", str(e))
            return None, None
        
    def corrector_mutators(
        self,
        mod,
        available_mutators: List[str],
        llm_bucket: List[str],
        historical_perf: Optional[str] = None,
        available_mutator_probs: Optional[Dict[str, float]] = None,
        corrector_model: Optional[str] = None,
        small_model_name: Optional[str] = None,
        small_mutators: Optional[List[str]] = None,
        small_next_model: Optional[str] = None,
        leaf_score: Optional[float] = None,
        small_child_score: Optional[float] = None,
        model_performance: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        if not corrector_model:
            raise ValueError("corrector_mutators called without corrector_model")

        self._call_seq += 1
        call_id = self._call_seq
        logger.warning("[LLM %d] corrector_mutators(): corrector_model=%s", call_id, corrector_model)

        system_prompt, user_prompt = self._build_correction_prompt(
            llm_bucket=llm_bucket,
            corrector_model=corrector_model,
            available_mutators=available_mutators,
            historical_perf=historical_perf,
            mutator_probs=available_mutator_probs,
            small_model_name=small_model_name or "UNKNOWN_SMALL_MODEL",
            small_mutators=small_mutators or [],
            small_next_model=small_next_model or "",
            leaf_score=leaf_score,
            small_child_score=small_child_score,
            model_performance=model_performance,
        )

        full_prompt = system_prompt + user_prompt
        p_chars, p_tokens = self._lengths(full_prompt)
        if self.verbose:
            logger.warning("\n Full Corrector Prompt \n%s", full_prompt)
        logger.warning("\n==== Current Corrector Prompt stats (chars=%d, tokens≈%d) ====\n", p_chars, p_tokens)

        self.cum_prompt_chars += p_chars
        self.cum_prompt_tokens += p_tokens

        try:
            t0 = time.perf_counter()

            if self._is_openai_model(corrector_model):
                response = self._openai_client.chat.completions.create(
                    model=corrector_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            elif self._is_google_model(corrector_model):
                response = self.nebius_client.chat.completions.create(
                    model=corrector_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            else:
                response = self.ns_client.chat.completions.create(
                    model=corrector_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

            api_latency = time.perf_counter() - t0

            content = response.choices[0].message.content
            logger.warning("[LLM %d] corrector API latency: %.3f sec", call_id, api_latency)

            a_chars, a_tokens = self._lengths(content)
            if self.verbose:
                logger.warning("\n Corrector raw response:\n%s", content)
            logger.warning("\n==== Current Corrector Answer stats (chars=%d, tokens≈%d) ====\n", a_chars, a_tokens)

            self.cum_answer_chars += a_chars
            self.cum_answer_tokens += a_tokens

            mutator_list, next_model = self._extract_mutators_list_and_model(
                model_text=content,
                valid_mutators=available_mutators,
                valid_models=llm_bucket,
            )
            logger.warning(
                "[LLM %d] corrector parsed output: mutators=%s next_model=%s",
                call_id, mutator_list, next_model
            )

            if not mutator_list:
                return None, next_model
            return mutator_list, next_model

        except Exception as e:
            logger.warning("Corrector ChatCompletion failed: %s", str(e))
            return None, None


    def _build_prompt(
        self,
        llm_bucket: List[str],
        current_model: str,
        available_mutators: List[str],
        historical_perf: Optional[str],
        mutator_probs: Optional[Dict[str, float]] = None,
        model_performance: Optional[str] = None,
    ) -> Tuple[str, str]:
        system_msg = (
            "You are an AI scheduling assistant integrated with TVM MetaSchedule. "
            "We are performing a Monte Carlo Tree Search (MCTS) to find an optimal "
            "schedule transformation sequence for a given IRModule. In this MCTS tree, "
            "the 'current schedule' is the leaf we are expanding, while 'immediate parent' "
            "and 'grandparent' refer to the ancestors in the tree. Each schedule has an IR, "
            "a sequence of transformations (the trace), and a predicted performance score "
            "from TVM's default XGBoost cost model.\n\n"

            "You are given:\n"
            " - The IRModule for the current schedule\n"
            " - Historical performance info summarizing the current schedule, its parent "
            "   and its grandparent schedules (their IR, traces, and predicted scores (do not "
            "   overly rely on predicted scores as they can be inaccurate sometimes))\n"
            " - A list of possible mutators (transformations) that can be applied next\n"

            " - Search context: leaf depth and trials progress (trial_count / max_trials)\n"
            " - Global per-model stats:\n"
            "       - hit_rate: fraction of calls where predicted score of a child (child produced by applying mutators generated by that model to its parent) is higher than predicted score of its parent\n"
            "       - number of times the model is called\n"
            "       - number of errors the model makes (+1 if invalid mutators, +1 if invalid next model name)\n"
            " - Local context:\n"
            "       - Model used to expand the current node and the model's number of parameters \n"
            "       - Model used to expand the parant node and the model's number of parameters \n"
            "       - Model used to expand the grandparent node and the model's number of parameters\n"

            "Please compare the IR, trace, and predicted scores of these schedules to see what "
            "changes might improve the current schedule performance. Then propose a *sequence* of transformations "
            "(one or more) from the provided list.\n"
            "This will allow MCTS to explore those transformations.\n"
            "NOTE: You may repeat the same mutator multiple times if you think it could be beneficial. "
            "For instance, 'meta_schedule.MutateTileSize(...)' might choose different tile sizes each time, "
            "so repeating it can explore a range of tiling configurations.\n"
            "IMPORTANT: If you choose one of the mutators, you MUST include the number in parentheses following the mutator\n"
            "If you omit the '(0x...)' part, your answer is invalid.\n"

            "Additionally, choose a single model from the provided model list to expand the current node's child node."
            "Smaller models are cheaper to run but might not produce as competitive results as larger models. \n"
            "We want to use the smallest model that could give the best results.\n"
            "Locally, consider the model used to expand the current node and the model's number of parameters, "
            "the model used to expand the parent node and the model's number of parameters, the model used to expand the grandparent node and the model's number of parameters, "
            " and the difference of predicted scores of current, parent and grandparent schedules.\n"
            "Globally, consider the current node's model hit rate, "
            "which is the percentage of times that the model leads to a better child compared with its parent. "
            "Make sure the largest model in the model bucket is called for at least twenty percent of the total number of model calls."
            "If some models have been called for only a very small number of times, make sure they also get called sufficiently, as we want to explore models with potential higher hit rates.\n"
            "Prefer models with a smaller number of errors. \n"

            "Output your answer in a single valid JSON object.\n"
            "Your answer should be in the exact following format \n"
            "{\n"
            '  "mutators": ["Fullname1", "Fullname2", "Fullname3", "Fullname4", "Fullname5", "Fullname6", "Fullname7", "Fullname8", "Fullname9", "Fullname10"...],\n'
            '  "next_model": "...",\n'
            "}\n"

        )
        user_msg = (
                "=== Historical Performance Info (Leaf, Parent, Grandparent) ===\n"
                f"{historical_perf}\n\n")
        user_msg += (
            "=== Available Mutators ===\n"
            f"{available_mutators}\n\n"
        )

        if model_performance:
            user_msg += model_performance.rstrip() + "\n\n"
            
        return system_msg, user_msg
    
    def _build_correction_prompt(
        self,
        llm_bucket: List[str],
        corrector_model: str,
        available_mutators: List[str],
        historical_perf: Optional[str],
        mutator_probs: Optional[Dict[str, float]] = None,
        small_model_name: str = "",
        small_mutators: Optional[List[str]] = None,
        small_next_model: str = "",
        leaf_score: Optional[float] = None,
        small_child_score: Optional[float] = None,
        model_performance: Optional[str] = None,
    ) -> Tuple[str, str]:
        small_mutators = small_mutators or []

        system_msg = (
            "You are the LARGE-MODEL corrector in a TVM MetaSchedule MCTS search.\n"
            "A SMALL model has proposed a sequence of mutators and a next_model to use for expanding the child node.\n"
            "The SMALL model's proposal that triggered this correction call led to a predicted regression "
            "(predicted child score < predicted current score) according to the cost model.\n\n"
            "Your job:\n"
            "  1) Review the small model proposal using the SAME schedule context (IR, trace, predicted scores).\n"
            "  2) MODIFY the SMALL models answer by proposing new mutators, suggesting a different model name or both.\n"
            "  3) Output the FINAL mutator list and FINAL next_model.\n\n"
            "Constraints:\n"
            "  - Your output MUST be a single valid JSON object.\n"
            "  - 'mutators' must use exact strings from the Available Mutators list, INCLUDING the '(0x...)' suffix.\n"
            "  - 'next_model' must be exactly one of the provided model names in the model bucket.\n\n"
            "Important:\n"
            "  - This is a CORRECTION call and is separate from regular expansion calls.\n"
            "  - Predicted scores can be noisy; use IR/trace reasoning too.\n\n"
            "Return JSON in EXACT format:\n"
            "{\n"
            '  \"mutators\": [\"Fullname1\", \"Fullname2\", \"...\"],\n'
            '  \"next_model\": \"...\",\n'
            "}\n"
        )

        user_msg = "=== Historical Performance Info (Leaf, Parent, Grandparent) ===\n"
        user_msg += f"{historical_perf}\n\n"

        user_msg += "=== Available Mutators ===\n"
        user_msg += f"{available_mutators}\n\n"
        user_msg += "=== Small Model Proposal To Correct ===\n"
        user_msg += f"small_model_name: {small_model_name}\n"
        user_msg += f"proposed_mutators: {small_mutators}\n"
        user_msg += f"proposed_next_model: {small_next_model}\n"

        if (leaf_score is not None) and (small_child_score is not None):
            user_msg += (
                f"predicted_current_score (leaf): {leaf_score}\n"
                f"predicted_child_score_from_small_proposal: {small_child_score}\n"
                "Reason for correction: predicted_child_score_from_small_proposal < predicted_current_score\n"
            )
        user_msg += "\n"

        if model_performance:
            user_msg += model_performance.rstrip() + "\n\n"

        return system_msg, user_msg

    

    def _strip_wrapping_value(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"^\s*(\*\*|__)", "", s)
        s = re.sub(r"(\*\*|__)\s*$", "", s)
        s = s.strip().strip("`").strip()
        s = re.sub(r"[,\.\s]+$", "", s).strip()
        if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
            s = s[1:-1].strip()
        s = re.sub(r"[,\.\s]+$", "", s).strip()
        return s


    def _extract_balanced_block(self, s: str, open_ch: str, close_ch: str) -> str:
        if not s:
            return ""
        start = s.find(open_ch)
        if start == -1:
            return ""

        depth = 0
        in_str = False
        quote = None
        escape = False

        for i in range(start, len(s)):
            ch = s[i]

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                    quote = None
                continue
            if ch in ("'", '"'):
                in_str = True
                quote = ch
                continue

            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return s[start:]


    def _extract_last_square_list_containing_mutate(self, text: str) -> str:
        text = self._strip_code_fences(text or "")

        blocks: List[str] = []
        depth = 0
        start = None

        in_str = False
        quote = None
        escape = False

        for i, ch in enumerate(text):
            if depth == 0:
                if ch == "[":
                    start = i
                    depth = 1
                    in_str = False
                    quote = None
                    escape = False
                continue

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                    quote = None
                continue

            if ch in ("'", '"'):
                in_str = True
                quote = ch
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    block = text[start : i + 1]
                    low = block.lower()
                    if ("mutate" in low) or ("meta_schedule" in low):
                        blocks.append(block)
                    start = None

        return blocks[-1].strip() if blocks else ""


    def _infer_next_model_from_text(self, text: str, valid_models: List[str]) -> str:
        if not text:
            return ""
        low = text.lower()

        best = ""
        best_pos = -1

        for m in valid_models:
            pos = low.rfind(m.lower())
            if pos > best_pos:
                best_pos = pos
                best = m

        for m in valid_models:
            if "/" in m:
                tail = m.split("/", 1)[1]
                pos = low.rfind(tail.lower())
                if pos > best_pos:
                    best_pos = pos
                    best = m

        return best


    def _scan_for_mutators_anywhere(
        self,
        text: str,
        alias_to_full: Dict[str, List[str]],
        full_to_core_norm: Dict[str, str],
    ) -> List[str]:
        text = self._strip_code_fences(text or "")

        # Find MutateSomething tokens in order
        pat = re.compile(r"(?:meta_schedule\.)?(Mutate[A-Za-z0-9_]+)", re.IGNORECASE)

        out: List[str] = []
        for m in pat.finditer(text):
            core = m.group(1)  # e.g., MutateTileSize
            a = self._norm(core)
            fulls = alias_to_full.get(a, [])
            if not fulls:
                continue
            if len(fulls) == 1:
                out.append(fulls[0])
            else:
                # disambiguate by similarity
                seg_norm = self._norm(core)
                best_full = None
                best_sim = -1.0
                for f in fulls:
                    core_norm = full_to_core_norm.get(f, "")
                    sim = difflib.SequenceMatcher(None, seg_norm, core_norm).ratio()
                    if sim > best_sim:
                        best_sim = sim
                        best_full = f
                out.append(best_full or fulls[0])

        return out

    
    def _strip_code_fences(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(
            r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)\s*```",
            r"\1",
            text,
            flags=re.IGNORECASE,
        ).strip()
    
    def _parse_jsonish_dict(self, s: str) -> Optional[dict]:
        if not s:
            return None

        cand = s.strip()
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        cand2 = cand
        cand2 = re.sub(r"/\*[\s\S]*?\*/", "", cand2)
        cand2 = re.sub(r"(?m)^\s*//.*$", "", cand2)
        cand2 = self._strip_hash_comments(cand2)
        cand2 = re.sub(r",\s*([}\]])", r"\1", cand2)

        try:
            obj = json.loads(cand2)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        py = cand2
        py = re.sub(r"(?<![A-Za-z0-9_])null(?![A-Za-z0-9_])", "None", py, flags=re.IGNORECASE)
        py = re.sub(r"(?<![A-Za-z0-9_])true(?![A-Za-z0-9_])", "True", py, flags=re.IGNORECASE)
        py = re.sub(r"(?<![A-Za-z0-9_])false(?![A-Za-z0-9_])", "False", py, flags=re.IGNORECASE)

        try:
            obj = ast.literal_eval(py)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        return None




    def _extract_first_json_object(self, text: str) -> Optional[dict]:
        if not text:
            return None

        text = self._strip_code_fences(text)
        m = re.search(r"(?is)<json>\s*(\{.*?\})\s*</json>", text)
        if m:
            obj = self._parse_jsonish_dict(m.group(1))
            if isinstance(obj, dict):
                return obj

        objs: List[str] = []

        depth = 0
        start = None

        in_str = False
        quote = None
        escape = False

        for i, ch in enumerate(text):
            if depth == 0:
                if ch == "{":
                    start = i
                    depth = 1
                    # reset string state for this object
                    in_str = False
                    quote = None
                    escape = False
                continue

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                    quote = None
                continue

            # not in a string (but inside object)
            if ch in ('"', "'"):
                in_str = True
                quote = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start : i + 1])
                    start = None
                    # reset string state
                    in_str = False
                    quote = None
                    escape = False

        for cand in reversed(objs):
            obj = self._parse_jsonish_dict(cand)
            if isinstance(obj, dict):
                return obj

        return None

    def _strip_hash_comments(self, s: str) -> str:
        if not s:
            return s

        out = []
        in_str = False
        quote = None
        escape = False

        i = 0
        while i < len(s):
            ch = s[i]

            if in_str:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                    quote = None
                i += 1
                continue

            # not in string
            if ch in ('"', "'"):
                in_str = True
                quote = ch
                out.append(ch)
                i += 1
                continue

            if ch == "#":
                # skip to end of line
                while i < len(s) and s[i] not in "\n\r":
                    i += 1
                continue

            out.append(ch)
            i += 1

        return "".join(out)



    def _json_get(self, d: dict, *keys: str) -> Optional[object]:
        for k in keys:
            if k in d:
                return d[k]
        return None


    def _coerce_mutators_to_text(self, mut_val: object) -> str:
        if mut_val is None:
            return ""

        # If model returns just a number, we can't infer which mutators
        if isinstance(mut_val, (int, float)):
            return ""

        if isinstance(mut_val, str):
            return mut_val.strip()

        if isinstance(mut_val, list):
            parts: List[str] = []
            for x in mut_val:
                if x is None:
                    continue
                if isinstance(x, str):
                    parts.append(x.strip())
                elif isinstance(x, dict):
                    name = x.get("name") or x.get("mutator") or x.get("value")
                    if name:
                        parts.append(str(name).strip())
                else:
                    parts.append(str(x).strip())
            return "\n".join([p for p in parts if p])

        return str(mut_val).strip()


    def _norm(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def _mutator_core_name(self, full: str) -> str:
        base = (full or "").split(".")[-1]
        base = re.sub(r"\(.*\)\s*$", "", base)  # strip trailing '(...)'
        return base.strip()

    def _camel_tokens(self, name: str) -> List[str]:
        toks = re.findall(r"[A-Z][a-z0-9]*", name or "")
        return toks or ([name] if name else [])

    def _build_mutator_alias_index(
        self, valid_mutators: List[str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        alias_to_full: Dict[str, List[str]] = {}
        full_to_core_norm: Dict[str, str] = {}

        # Count first-token frequencies to avoid ambiguous short aliases
        first_tok_freq: Dict[str, int] = {}
        per_full_tokens: Dict[str, List[str]] = {}

        for full in valid_mutators:
            core = self._mutator_core_name(full)           # MutateTileSize
            full_to_core_norm[full] = self._norm(core)

            toks = self._camel_tokens(core)                # ['Mutate','Tile','Size']
            toks2 = toks[1:] if toks and toks[0].lower() == "mutate" else toks
            per_full_tokens[full] = toks2

            if toks2:
                first = toks2[0].lower()
                first_tok_freq[first] = first_tok_freq.get(first, 0) + 1

        def add_alias(alias: str, full: str) -> None:
            a = self._norm(alias)
            if len(a) < 4:
                return
            alias_to_full.setdefault(a, []).append(full)

        for full in valid_mutators:
            core = self._mutator_core_name(full)           # MutateTileSize
            toks2 = per_full_tokens.get(full, [])          # ['Tile','Size']

            # Always allow matching the core forms
            add_alias(core, full)                          # MutateTileSize
            if core.lower().startswith("mutate"):
                add_alias(core[6:], full)                  # TileSize

            if toks2:
                joined = "".join(toks2)                    # TileSize / ComputeLocation
                add_alias(joined, full)
                add_alias("_".join(toks2), full)           # Tile_Size / Compute_Location
                add_alias(" ".join(toks2), full)           # "Tile Size" / "Compute Location"

                # Add a short "prefix token" alias ONLY if it's short (<=4) AND unique.
                # This is what enables "Tile" -> MutateTileSize, while avoiding generic "Compute".
                first = toks2[0]
                if len(first) <= 4 and first_tok_freq.get(first.lower(), 0) == 1:
                    add_alias(first, full)                 # Tile

        return alias_to_full, full_to_core_norm

    def _extract_mutators_block(
        self,
        model_text: str,
        alias_to_full: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        text = self._strip_code_fences(model_text or "")
        lines = text.splitlines()

        # Match "mutators", "mutator_list", etc with optional quotes/backticks, and ':' or '='
        key_re = re.compile(
            r"(?ix)"
            r"(?:^|[^\w])"  # start or non-word
            r"[`\"']?\s*"
            r"(mutators?|mutator_list|actions|transforms?)"
            r"\s*[`\"']?\s*"
            r"[:=]\s*"
            r"(.*)$"
        )

        # Stop collecting if we hit other known keys
        stop_re = re.compile(
            r"(?ix)^\s*"
            r"[`\"']?\s*"
            r"(next[_\s-]*model|nextmodel|next\s*model|mutator[_\s-]*rationale|next[_\s-]*model[_\s-]*rationale)"
            r"\s*[`\"']?\s*[:=]"
        )

        list_prefix_re = re.compile(r"^\s*([-*•]|\d+[\).])\s*")
        heading_re = re.compile(r"^\s*#{1,6}\s*")

        def _is_mutatorish_line(s: str) -> bool:
            if not s:
                return False
            low = s.lower()
            if "mutate" in low or "meta_schedule" in low or "0x" in low:
                return True
            if alias_to_full:
                s_norm = self._norm(s)
                for a in alias_to_full.keys():
                    if a and a in s_norm:
                        return True
            return False

        for i, raw in enumerate(lines):
            if not raw.strip():
                continue

            line = heading_re.sub("", raw.strip())
            line = list_prefix_re.sub("", line)

            m = key_re.search(line)
            if not m:
                continue

            tail = (m.group(2) or "").strip()
            rest = tail + "\n" + "\n".join(lines[i + 1 :])
            if "[" in rest:
                cand = rest[rest.find("[") :]
                lst = self._extract_balanced_block(cand, "[", "]")
                if lst and ("mutate" in lst.lower() or "meta_schedule" in lst.lower()):
                    return lst.strip()
            if tail:
                return tail.strip()
            collected: List[str] = []
            for j in range(i + 1, len(lines)):
                s = heading_re.sub("", lines[j].strip())
                s = list_prefix_re.sub("", s)
                if not s:
                    continue
                if stop_re.search(s):
                    break
                if _is_mutatorish_line(s):
                    collected.append(s)
                    continue
                break

            return "\n".join(collected).strip()

        return ""



    def _extract_next_model_value(self, model_text: str) -> str:
        text = self._strip_code_fences(model_text or "")
        lines = text.splitlines()

        key_re = re.compile(
            r"(?ix)"
            r"(?:^|[^\w])"
            r"[`\"']?\s*(next[_\s-]*model|nextmodel|next\s*model)\s*[`\"']?\s*"
            r"[:=]\s*(.*)$"
        )

        list_prefix_re = re.compile(r"^\s*([-*•]|\d+[\).])\s*")
        heading_re = re.compile(r"^\s*#{1,6}\s*")

        for raw in lines:
            if not raw.strip():
                continue
            line = heading_re.sub("", raw.strip())
            line = list_prefix_re.sub("", line)

            m = key_re.search(line)
            if not m:
                continue

            val = self._strip_wrapping_value(m.group(2) or "")
            # model names won't contain spaces; if there's prose after it, drop it
            if " " in val:
                val = val.split()[0]
            return val

        return ""



    def _match_model_fuzzy(self, candidate: str, valid_models: List[str]) -> Optional[str]:
        """
        More forgiving model matching:
          - case-insensitive exact
          - match by aliases (drop org prefix, drop leading 'Meta-' etc.)
          - difflib fallback
        """
        if not candidate:
            return None

        cand = candidate.strip().strip("`\"'")  # strip code ticks/quotes
        if not cand:
            return None

        # 1) case-insensitive exact
        for m in valid_models:
            if cand.lower() == m.lower():
                return m

        cand_norm = self._norm(cand)
        if not cand_norm:
            return None

        alias_to_models: Dict[str, List[str]] = {}
        model_norm: Dict[str, str] = {}

        def add(alias: str, model: str) -> None:
            a = self._norm(alias)
            if len(a) < 4:
                return
            alias_to_models.setdefault(a, []).append(model)

        for m in valid_models:
            model_norm[m] = self._norm(m)
            add(m, m)
            if "/" in m:
                tail = m.split("/", 1)[1]
                add(tail, m)
                tail2 = re.sub(r"(?i)^meta[-_ ]*", "", tail)
                add(tail2, m)
                tail3 = re.sub(r"(?i)instruct", "", tail2)
                add(tail3, m)
        hits: List[Tuple[int, str, List[str]]] = []
        for a, ms in alias_to_models.items():
            if a and a in cand_norm:
                hits.append((len(a), a, ms))
        if hits:
            hits.sort(reverse=True)
            ms = hits[0][2]
            if len(ms) == 1:
                return ms[0]
            best_m = None
            best_sim = -1.0
            for m in ms:
                sim = difflib.SequenceMatcher(None, cand_norm, model_norm.get(m, "")).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_m = m
            return best_m or ms[0]
        best_m = None
        best_sim = 0.0
        for m in valid_models:
            sim = difflib.SequenceMatcher(None, cand_norm, model_norm.get(m, "")).ratio()
            if sim > best_sim:
                best_sim = sim
                best_m = m
        return best_m if (best_m and best_sim >= 0.55) else None

    def _extract_mutators_list_and_model(
        self,
        model_text: str,
        valid_mutators: List[str],
        valid_models: List[str],
    ) -> Tuple[List[str], Optional[str]]:
        chosen_mutators: List[str] = []
        next_model: Optional[str] = None
    
        alias_to_full, full_to_core_norm = self._build_mutator_alias_index(valid_mutators)
        payload = self._extract_first_json_object(model_text)
        mut_block = ""
        nm_val = ""
    
        if isinstance(payload, dict):
            mut_val = self._json_get(payload, "mutators", "Mutators", "mutator_list", "actions")
            nm_raw = self._json_get(payload, "next_model", "nextModel", "NextModel", "model", "next")
    
            mut_block = self._coerce_mutators_to_text(mut_val)
            nm_val = (str(nm_raw).strip() if nm_raw is not None else "")
        if not mut_block:
            mut_block = self._extract_mutators_block(model_text, alias_to_full)
        if not nm_val:
            nm_val = self._extract_next_model_value(model_text)
        if not mut_block:
            mut_block = self._extract_last_square_list_containing_mutate(model_text)
        if not nm_val:
            nm_val = self._infer_next_model_from_text(model_text, valid_models)
        if mut_block:
            segments = re.split(r"(?:,|;|\n|\r|\s*->\s*|\s*\|\s*)+", mut_block)
            for seg in segments:
                seg = (seg or "").strip()
                if not seg:
                    continue
                
                seg_norm = self._norm(seg)
                if not seg_norm:
                    continue
                
                hits: List[Tuple[int, int, str]] = []
                for alias_norm in alias_to_full.keys():
                    start = 0
                    while True:
                        idx = seg_norm.find(alias_norm, start)
                        if idx == -1:
                            break
                        hits.append((idx, idx + len(alias_norm), alias_norm))
                        start = idx + 1
    
                if hits:
                    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    
                    picked: List[Tuple[int, int, str]] = []
                    cur_end = -1
                    for st, en, a in hits:
                        if st < cur_end:
                            continue
                        picked.append((st, en, a))
                        cur_end = en
    
                    for _, _, a in picked:
                        fulls = alias_to_full.get(a, [])
                        if not fulls:
                            continue
                        if len(fulls) == 1:
                            chosen_mutators.append(fulls[0])
                        else:
                            best_full = None
                            best_sim = -1.0
                            for f in fulls:
                                core_norm = full_to_core_norm.get(f, "")
                                sim = difflib.SequenceMatcher(None, seg_norm, core_norm).ratio()
                                if sim > best_sim:
                                    best_sim = sim
                                    best_full = f
                            chosen_mutators.append(best_full or fulls[0])
                else:
                    best_full = None
                    best_sim = 0.0
                    for f, core_norm in full_to_core_norm.items():
                        sim = difflib.SequenceMatcher(None, seg_norm, core_norm).ratio()
                        if sim > best_sim:
                            best_sim = sim
                            best_full = f
                    if best_full and best_sim >= 0.55:
                        chosen_mutators.append(best_full)
                    else:
                        logger.warning("Could not match mutator segment from LLM: '%s'", seg)
        if nm_val:
            next_model = self._match_model_fuzzy(nm_val, valid_models)
            if next_model is None:
                inferred = self._infer_next_model_from_text(model_text, valid_models)
                next_model = self._match_model_fuzzy(inferred, valid_models) if inferred else None
        if not chosen_mutators:
            chosen_mutators = self._scan_for_mutators_anywhere(model_text, alias_to_full, full_to_core_norm)
    
        return chosen_mutators, next_model