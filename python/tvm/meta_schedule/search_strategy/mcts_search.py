import math
import random
import logging
import heapq
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
    Dict,
    Optional,
    Set,
)

import tvm
from tvm._ffi import get_global_func
from tvm.runtime import Object
from tvm.tir import Schedule
from tvm.tir.schedule import Trace
from tvm.ir import IRModule

from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.runner import RunnerResult
from .search_strategy import SearchStrategy
from .search_strategy import PySearchStrategy
from .search_strategy import MeasureCandidate
from ..postproc import Postproc
from ..mutator import Mutator
from ..database import Workload
from .. import _ffi_api
from .llm_guidance import LLMGuidancePolicy

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext

from ..database import TuningRecord

try:
    from tvm.error import InvalidScheduleError
except ImportError:
    InvalidScheduleError = tvm.TVMError

logger = logging.getLogger("meta_schedule")
logger.setLevel(logging.DEBUG)

@dataclass
class LLMConfig:
    api_model_name: str
    description: str = ""
    relative_cost: float = 1.0

@dataclass
class ModelStats:
    n_calls: int = 0
    total_latency: float = 0.0
    n_scored: int = 0
    n_hits: int = 0
    total_tokens: int = 0 # not used
    n_errors: int = 0
    n_corrector_calls: int = 0
    corrector_total_latency: float = 0.0
    n_corrector_scored: int = 0
    n_corrector_hits: int = 0
    n_corrector_errors: int = 0
    param_count: Optional[float] = None
    phi_small: Optional[float] = None


class _SizedMinHeap:
    # Not used
    def __init__(self, size_limit: int):
        self._size_limit = size_limit
        self._heap = []
        self._push_counter = 0

    def push(self, sch: Schedule, score: float, measured_flag: bool) -> None:
        neg_score = -score
        self._push_counter += 1
        item = (neg_score, self._push_counter, sch, measured_flag)
        if len(self._heap) < self._size_limit:
            heapq.heappush(self._heap, item)
        else:
            worst_neg, _, _, _ = self._heap[0]
            if neg_score > worst_neg:
                return
            heapq.heapreplace(self._heap, item)

    def items_descending(self) -> List[Tuple[float, Schedule, bool]]:
        items = []
        for (neg, _, sch, meas) in self._heap:
            score = -neg
            items.append((score, sch, meas))
        items.sort(key=lambda x: x[0], reverse=True)
        return items


class MCTSNode:
    __slots__ = [
        "schedule",
        "parent",
        "children",
        "visits",
        "total_value",
        "depth",
        "model_name",
        "model_call_count",
        "model_cumulative_latency",
        "model_cumulative_token_cost",
        "expand_failures",
        "is_terminal",
    ]

    def __init__(
        self,
        schedule: Optional[Schedule],
        parent: Optional["MCTSNode"],
        depth: int,
    ):
        self.schedule = schedule
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.total_value = 0.0
        self.depth = depth

        # Added four more fields
        self.model_name: Optional[str] = None
        self.model_call_count: int = 0
        self.model_cumulative_latency: float = 0.0
        self.model_cumulative_token_cost: float = 0.0

        self.expand_failures: int = 0
        self.is_terminal: bool = False


    def clone_tree(self) -> "MCTSNode":
        """
        Recursively clones this node (and its sub-tree).
        """
        new_node = MCTSNode(self.schedule, None, self.depth)
        new_node.visits = self.visits
        new_node.total_value = self.total_value

        # Added four more fields
        new_node.model_name = self.model_name
        new_node.model_call_count = self.model_call_count
        new_node.model_cumulative_latency = self.model_cumulative_latency
        new_node.model_cumulative_token_cost = self.model_cumulative_token_cost
        new_node.expand_failures = self.expand_failures
        new_node.is_terminal = self.is_terminal

        for ch in self.children:
            child_copy = ch.clone_tree()
            child_copy.parent = new_node
            new_node.children.append(child_copy)
        return new_node


class MCTSTuner:

    def __init__(
        self,
        population_size: int,
        init_measured_ratio: float,
        init_min_unmeasured: int,
        max_fail_count: int,
        genetic_num_iters: int,
        genetic_mutate_prob: float,
        genetic_max_fail_count: int,
        num_empty_iters_before_early_stop: int,
        max_stale_iters: int,
        diversity_epsilon: float,
        max_stale_diversity_iters: int,
        trace_commit: bool,
        verbose: int,
        mcts_ucb_constant: float,
        mcts_max_depth: Optional[int],
        mcts_num_threads: int,
        mcts_num_rollouts_per_expansion: int,
        postprocs: List[Postproc],
        mutator_probs: Dict[Mutator, float],
        context: "TuneContext",
        cost_model: Optional["CostModel"],
        database: Optional["Database"],
        workload_key: Optional[Workload],
        use_llm: bool,
        llm_budget: int,
        llm_policy: Optional["LLMGuidancePolicy"] = None,
        llm_bucket: Optional[List[str]] = None,
        default_llm_model_name: Optional[str] = None,
        mcts_model_balance_lambda: float = 0.0,
        llm_param_count: Optional[Dict[str, float]] = None,
    ):
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob
        self.genetic_max_fail_count = genetic_max_fail_count
        self.num_empty_iters_before_early_stop = num_empty_iters_before_early_stop
        self.max_stale_iters = max_stale_iters
        self.diversity_epsilon = diversity_epsilon
        self.max_stale_diversity_iters = max_stale_diversity_iters
        self.trace_commit = trace_commit
        self.verbose = verbose
        self.mcts_ucb_constant = mcts_ucb_constant
        self.mcts_max_depth = mcts_max_depth
        self.mcts_num_threads = mcts_num_threads
        self.mcts_num_rollouts_per_expansion = mcts_num_rollouts_per_expansion
        self._postprocs = postprocs
        self._mutator_probs = mutator_probs
        self._ctx = context
        self._cost_model = cost_model
        self._database = database
        self._workload_key = workload_key

        self._workload_cache: Dict[int, Workload] = {}
        self._mutator_failure_count: Dict[object, int] = {"total": 0}
        self._search_state: Optional["MCTSTuningState"] = None

        self.use_llm = use_llm
        self.llm_budget = llm_budget
        self.llm_policy = llm_policy
        self.default_llm_model_name = default_llm_model_name
        self.llm_bucket: List[str] = list(llm_bucket or [])
        self.mcts_model_balance_lambda = float(mcts_model_balance_lambda)
        self._llm_model_stats: Dict[str, ModelStats] = {
            m: ModelStats() for m in self.llm_bucket
        }
        self._init_llm_param_and_phi(llm_param_count or {})
        self._small_model_regression_streak: int = 0

        # Correct on every K-th regression in the streak: 5,10,15,...
        self._corrector_every_k_regressions: int = 2
        self.mcts_dead_end_fail_limit = 5

    def attach_search_state(self, search_state: "MCTSTuningState") -> None:
        self._search_state = search_state

    def _init_llm_param_and_phi(self, llm_param_count: Dict[str, float]) -> None:
        if not self.llm_bucket:
            return
        max_known = max(llm_param_count.values()) if llm_param_count else 1.0

        logs: Dict[str, float] = {}
        for m in self.llm_bucket:
            st = self._llm_model_stats.setdefault(m, ModelStats())

            pc = float(llm_param_count.get(m, max_known))
            if pc <= 0:
                pc = max_known

            st.param_count = pc
            logs[m] = math.log(pc)

        log_max = max(logs.values())
        log_min = min(logs.values())
        denom = (log_max - log_min) + 1e-12  # ε

        for m in self.llm_bucket:
            st = self._llm_model_stats[m]
            phi = (log_max - logs[m]) / denom
            # clamp into [0,1] (numerical safety)
            st.phi_small = max(0.0, min(1.0, float(phi)))

        logger.warning(
            "[MA-UCT] lambda=%.4f, phi_small=%s",
            self.mcts_model_balance_lambda,
            {m: self._llm_model_stats[m].phi_small for m in self.llm_bucket},
        )

    def _phi_small(self, model_name: Optional[str]) -> float:
        if not model_name:
            return 0.0
        st = self._llm_model_stats.get(model_name)
        if st is None or st.phi_small is None:
            return 0.0
        return float(st.phi_small)
    
    def _is_large_model(self, model_name: Optional[str]) -> bool:
        return (
            model_name is not None
            and self.default_llm_model_name is not None
            and model_name == self.default_llm_model_name
        )

    def _is_small_model(self, model_name: Optional[str]) -> bool:
        return (
            model_name is not None
            and model_name in self.llm_bucket
            and self.default_llm_model_name is not None
            and model_name != self.default_llm_model_name
        )

    def _alt_model(self) -> Optional[str]:
        m = self.default_llm_model_name
        if m and (m in self.llm_bucket):
            return m
        return None

    def explore(
        self,
        mcts_root: MCTSNode,
        population: List[Tuple[tvm.tir.Schedule, bool]],
        dynamic_pop_size: int,
        rand_state: int,
    ) -> List[Tuple[tvm.tir.Schedule, bool]]:
        """
        Perform expansions (self.genetic_num_iters times) from mcts_root,
        in each iteration generate up to `dynamic_pop_size` new children.
        Then, gather all nodes in the tree, returning them as (schedule, measured_flag).
        We'll let generate_measure_candidates() do the actual eps-greedy picking.
        """
        logger.warning(
            "[DEBUG] explore() called with dynamic_pop_size=%d, genetic_num_iters=%d",
            dynamic_pop_size,
            self.genetic_num_iters
        )

        if not mcts_root or not mcts_root.children:
            logger.warning("explore(): Root is empty or has no children. Returning existing population.")
            return population

        total_expansions = 0
        for gen_iter in range(self.genetic_num_iters):
            new_children_count = 0
            fail_count = 0

            logger.warning("explore(): Starting generation %d ...", gen_iter)

            while new_children_count < dynamic_pop_size:
                leaf = self._select(mcts_root)
                if leaf is None:
                    logger.warning(
                        "explore(): MCTS: Leaf is None in selection => break expansions."
                    )
                    fail_count += 1 
                    if fail_count >= self.genetic_max_fail_count:
                        break
                    continue

                logger.warning(
                    "explore(): [gen=%d] Selected leaf node at depth=%d with %d children",
                    gen_iter, leaf.depth, len(leaf.children)
                )

                new_node = self._expand(leaf, rand_state)
                if new_node is None:
                    leaf.expand_failures += 1
    
                    if leaf.expand_failures >= self.mcts_dead_end_fail_limit:
                        leaf.is_terminal = True
                        logger.warning(
                            "[MCTS] Marking node as terminal (dead-end for expansion): depth=%d visits=%d children=%d expand_failures=%d",
                            leaf.depth, leaf.visits, len(leaf.children), leaf.expand_failures
                        )

                    self._backprop(leaf, 0.0)
                    fail_count += 1
                    logger.warning(
                        "explore(): Failed to expand leaf at depth=%d (fail_count=%d)",
                        leaf.depth, fail_count
                    )
                    if fail_count >= self.genetic_max_fail_count:
                        logger.warning(
                            "explore(): Too many expansion failures => break expansions."
                        )
                        break
                    continue

                leaf.expand_failures = 0
                leaf.is_terminal = False

                logger.warning(
                    "explore(): Successfully expanded leaf at depth=%d => new node at depth=%d; "
                    "now leaf has %d children total",
                    leaf.depth, new_node.depth, len(leaf.children)
                )
                value = self._simulate_node(new_node, rand_state)
                logger.warning(
                    "explore(): Simulation done for new node at depth=%d => value=%.4f",
                    new_node.depth, value
                )

                self._backprop(new_node, value)
                logger.warning(
                    "explore(): Backprop done => node visits=%d, total_value=%.4f",
                    new_node.visits, new_node.total_value
                )

                new_children_count += 1
                total_expansions += 1

            logger.warning(
                "explore(): [gen=%d] expansions=%d, fail_count=%d so far in this generation",
                gen_iter, new_children_count, fail_count
            )

        logger.warning(
            "explore(): All expansions complete => total_expansions=%d across %d generations.",
            total_expansions, self.genetic_num_iters
        )

        all_nodes = self._gather_tree_schedules(mcts_root)
        logger.warning(
            "explore(): Gathered %d total nodes (with schedules) from the MCTS tree.",
            len(all_nodes)
        )
        new_population = []
        for node in all_nodes:
            if node.schedule is not None:
                wl = self._commit_workload_cached(node.schedule)
                measured_flag = (wl is not None) and (wl in self._measured_workloads)
                new_population.append((node.schedule, measured_flag))

        logger.warning(
            "explore(): Returning a new population of size=%d (some may be measured).",
            len(new_population)
        )
        return new_population


    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        current = node
        while True:
            if current.parent is None:
                unvisited = [
                    c for c in current.children
                    if (c.visits == 0) and not (c.is_terminal and not c.children)
                ]
                if unvisited:
                    rng = getattr(self._search_state, "_rng", random)
                    current = rng.choice(unvisited)
                else:
                    current = self._select_by_ucb(current)

                if current is None:
                    return None
                continue  

            if current.visits == 0:
                return current
            
            max_children = 1 + int(0.5 * math.log2(current.visits + 1))

            if current.visits == 0 and not current.is_terminal:
                return current
            
            if len(current.children) < 2 and not current.is_terminal:
                return current
    
            next_child = self._select_by_ucb(current)
            if next_child is None:
                return None
            current = next_child

    def _select_by_ucb(self, node: MCTSNode) -> Optional[MCTSNode]:
        best_child = None
        best_key = (-1, -float("inf"))
        c = self.mcts_ucb_constant
        phis = [self._phi_small(ch.model_name) for ch in node.children]
        lam_eff = 0.0 if (max(phis) - min(phis) < 1e-12) else self.mcts_model_balance_lambda
        for ch in node.children:
            if ch.is_terminal and not ch.children:
                continue
            phi = self._phi_small(ch.model_name)
            if ch.visits == 0:
                score = phi
                key = (1, score)
            else:
                exploit = ch.total_value / ch.visits
                ratio = math.log(node.visits) / ch.visits
                explore = math.sqrt(ratio) if ratio > 0 else 0.0
                score = (1-lam_eff)*exploit + lam_eff*phi + c * explore
                key = (0, score)

            if key > best_key:
                best_key = key
                best_child = ch
        return best_child
    
    def _node_mean_latency(self, node: Optional[MCTSNode]) -> Optional[float]:
        if node is None or node.model_call_count <= 0:
            return None
        return node.model_cumulative_latency / node.model_call_count

    def _snapshot_llm_global_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        snap: Dict[str, Dict[str, Optional[float]]] = {}
        large = self.default_llm_model_name

        for m in self.llm_bucket:
            st = self._llm_model_stats.get(m, ModelStats())

            mean_latency = (st.total_latency / st.n_calls) if st.n_calls > 0 else None
            hit_rate = (st.n_hits / st.n_scored) if st.n_scored > 0 else None

            entry: Dict[str, Optional[float]] = {
                "mean_latency": mean_latency,
                "n_calls": float(st.n_calls),
                "hit_rate": hit_rate,
                "n_errors": float(st.n_errors),
                "param_count": st.param_count,
                "phi_small": st.phi_small,
            }

            if large and (m == large):
                corrector_mean_latency = (
                    (st.corrector_total_latency / st.n_corrector_calls)
                    if st.n_corrector_calls > 0 else None
                )
                corrector_hit_rate = (
                    (st.n_corrector_hits / st.n_corrector_scored)
                    if st.n_corrector_scored > 0 else None
                )
                entry.update({
                    "n_corrector_calls": float(st.n_corrector_calls),
                    "corrector_mean_latency": corrector_mean_latency,
                    "corrector_hit_rate": corrector_hit_rate,
                    "n_corrector_errors": float(st.n_corrector_errors),
                })

            snap[m] = entry

        return snap

    
    def _log_llm_bucket_call_counts(self, where: str = "") -> None:
        if not self.llm_bucket:
            logger.warning("[LLM] (no llm_bucket) %s", where)
            return

        large = self.default_llm_model_name

        lines: List[str] = []
        header = "[LLM] Call counts per model"
        if where:
            header += f" ({where})"
        lines.append(header)

        for m in self.llm_bucket:
            st = self._llm_model_stats.get(m, ModelStats())

            line = (
                f"  - {m}: regular_calls={int(st.n_calls)}, regular_call_errors={int(st.n_errors)}"
            )

            if large and (m == large):
                line += (
                    f", corrector_calls={int(st.n_corrector_calls)}, corrector_call_errors={int(st.n_corrector_errors)}"
                )

            lines.append(line)

        logger.warning("\n" + "\n".join(lines))


    
    def _best_global_model_by_hit_rate(self) -> Optional[str]:
        if not self.llm_bucket:
            return None

        best_model: Optional[str] = None
        best_hr: float = -1.0
        best_n: int = -1
        best_pc: float = float("inf")

        for m in self.llm_bucket:
            st = self._llm_model_stats.get(m)
            if st is None:
                continue
            if st.n_scored <= 0:
                continue

            hr = float(st.n_hits) / float(st.n_scored)
            pc = float(st.param_count) if st.param_count is not None else float("inf")

            better = False
            if hr > best_hr + 1e-12:
                better = True
            elif abs(hr - best_hr) <= 1e-12:
                if st.n_scored > best_n:
                    better = True
                elif st.n_scored == best_n and pc < best_pc - 1e-12:
                    better = True

            if better:
                best_model = m
                best_hr = hr
                best_n = st.n_scored
                best_pc = pc

        return best_model


    
    def _build_model_performance_string(self, leaf: MCTSNode) -> str:
        trial_count = int(getattr(self._search_state, "trial_count", 0)) if self._search_state else 0
        max_trials = int(getattr(self._search_state, "max_trials", 0)) if self._search_state else 0

        leaf_depth = leaf.depth
        parent_node = leaf.parent
        grandparent_node = parent_node.parent if parent_node else None
        grgrandparent_node = grandparent_node.parent if grandparent_node else None

        parent_model = parent_node.model_name if parent_node else None
        grandparent_model = grandparent_node.model_name if grandparent_node else None
        grgrandparent_model = grgrandparent_node.model_name if grgrandparent_node else None
        global_model_stats = self._snapshot_llm_global_stats()

        def _fmt_param_count(model_name: Optional[str]) -> str:
            if not model_name:
                return "N/A"
            st = global_model_stats.get(model_name, {})
            pc = st.get("param_count", None)
            if pc is None:
                return "N/A"
            try:
                return f"{float(pc) / 1e9:.1f}B"
            except Exception:
                return "N/A"

        parent_pc_str = _fmt_param_count(parent_model)
        grandparent_pc_str = _fmt_param_count(grandparent_model)
        grgrandparent_pc_str = _fmt_param_count(grgrandparent_model)

        lines: List[str] = []
        lines.append("=== Search Context ===")
        lines.append(f"Leaf depth: {leaf_depth}")
        lines.append(f"Trials progress: {trial_count} / {max_trials}")
        lines.append(f"Hit-rate definition: pred(child) > pred(parent)")
        lines.append("Number of errors definition: +1 if model generated invalid mutators, +1 if model generated invalid next_model name (per LLM call)")
        lines.append("")

        lines.append("=== Global Per-Model Stats (bucket-wide) ===")
        if global_model_stats and self.llm_bucket:
            large = self.default_llm_model_name

            for m in self.llm_bucket:
                st = global_model_stats.get(m, {})

                param_count = st.get("param_count", None)
                number_calls = st.get("n_calls", None)
                hit_rate = st.get("hit_rate", None)
                n_errors = st.get("n_errors", None)

                hit_rate_str = "N/A" if hit_rate is None else f"{hit_rate:.3f}"
                pc_str = "N/A" if param_count is None else f"{float(param_count)/1e9:.1f}B"
                number_calls_str = "N/A" if number_calls is None else f"{int(number_calls)}"
                n_errors_str = "N/A" if n_errors is None else f"{int(n_errors)}"

                line = (
                    f"Model {m}: params={pc_str}, "
                    f"regular_calls={number_calls_str}, regular_hit_rate={hit_rate_str}, regular_errors={n_errors_str}"
                )

                # Only show corrector stats for the default/large model
                if large and (m == large):
                    n_corrector_calls = st.get("n_corrector_calls", None)
                    corrector_hit_rate = st.get("corrector_hit_rate", None)
                    n_corrector_errors = st.get("n_corrector_errors", None)

                    n_corrector_calls_str = "N/A" if n_corrector_calls is None else f"{int(n_corrector_calls)}"
                    corrector_hit_rate_str = "N/A" if corrector_hit_rate is None else f"{corrector_hit_rate:.3f}"
                    n_corrector_errors_str = "N/A" if n_corrector_errors is None else f"{int(n_corrector_errors)}"

                    line += (
                        f"; corrector_calls={n_corrector_calls_str}, corrector_hit_rate={corrector_hit_rate_str}, corrector_errors={n_corrector_errors_str}"
                    )

                lines.append(line)
        else:
            lines.append("(no global model stats yet)")
        lines.append("")

        lines.append("=== Local Model Context (node-local) ===")
        lines.append(f"Model used to expand the current node: {parent_model or 'N/A'}; number of parameters: {parent_pc_str}")
        lines.append(f"Model used to expand the parent node: {grandparent_model or 'N/A'}; number of parameters: {grandparent_pc_str}")
        lines.append(f"Model used to expand the grandparent node: {grgrandparent_model or 'N/A'}; number of parameters: {grgrandparent_pc_str}")
        lines.append("Corrector stats (corrector_calls/corrector_hit_rate/corrector_errors) exist only for the largest model and are separate from regular expansion calls.")

        
        return "\n".join(lines)


    def _expand(self, leaf: MCTSNode, rand_state: int) -> Optional[MCTSNode]:
        if len(leaf.children) >= 2:
            return None
        if self.mcts_max_depth is not None and leaf.depth >= self.mcts_max_depth:
            return None
        if leaf.schedule is None:
            return None
        can_use_llm = (
            self.use_llm
            and self.llm_policy is not None
            and self.llm_budget > 0
            and (2 <= leaf.depth)
            and leaf.model_name is not None # could remove this later
            # and (len(leaf.children) == 0)
            # and (3<= leaf.depth <= 6 or 72 <= leaf.depth <= 76)
            # and (3<= leaf.depth <= 5 or len(leaf.children) == 0)
        )

        def _random_expand() -> Optional[MCTSNode]:
            logger.warning(
                "Not using LLM (disabled / no budget / no bucket / fallback). Using random mutator."
            )
            new_sch = self._try_mcts_mutation(leaf.schedule, rand_state)
            if not new_sch:
                logger.warning("Random mutation failed to produce a new schedule.")
                return None
            child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
            # Propagate the same model name down the tree (if any)
            child.model_name = leaf.model_name
            leaf.children.append(child)
            return child

        if not can_use_llm:
            return _random_expand()
        
        else:
            logger.warning("LLM usage is enabled. Gathering historical info for leaf, parent, and grandparent schedules.")
            new_sch = None
            historical_perf_parts = []
            short_hist_perf_parts = []
            leaf_score: float = 0.0
            try:
                # --- Current schedule ---
                leaf_score_list = self._predict_normalized_score([leaf.schedule])
                leaf_score = leaf_score_list[0] if leaf_score_list else 0.0
                try:
                    leaf_mod_str = leaf.schedule.mod.script()
                except Exception:
                    leaf_mod_str = "<failed to script IR>"
                leaf_trace_str = str(leaf.schedule.trace)
                historical_perf_parts.append(
                    "Current Schedule:\n"
                    f"Current Schedule's IR:\n{leaf_mod_str}\n\n"
                    f"Current Schedule's Trace:\n{leaf_trace_str}\n\n"
                    f"Current Schedule's Predicted Score by TVM's default cost model XGBoost: {leaf_score}\n"
                )
                short_hist_perf_parts.append(
                    "Current Schedule:\n"
                    f"Current Schedule's IR:\n{leaf_mod_str}\n\n"
                    f"Current Schedule's Predicted Score by TVM's default cost model XGBoost: {leaf_score}\n"
                )
    
                # --- Immediate parent schedule ---
                parent_node = leaf.parent
                if parent_node and parent_node.schedule is not None:
                    p1_sch = parent_node.schedule
                    scores_p1 = self._predict_normalized_score([p1_sch])
                    score_p1 = scores_p1[0] if scores_p1 else 0.0
    
                    try:
                        p1_mod_str = p1_sch.mod.script()
                    except Exception:
                        p1_mod_str = "<failed to script IR>"
                    p1_trace_str = str(p1_sch.trace)
                    historical_perf_parts.append(
                        "Immediate Parent Schedule:\n"
                        f"Immediate Parent's IR:\n{p1_mod_str}\n\n"
                        f"Immediate Parent's Trace:\n{p1_trace_str}\n\n"
                        f"Immediate Parent's Predicted Score by TVM's default cost model XGBoost: {score_p1}\n"
                    )
                    short_hist_perf_parts.append(
                        "Immediate Parent Schedule:\n"
                        f"Immediate Parent's IR:\n{p1_mod_str}\n\n"
                        f"Immediate Parent's Predicted Score by TVM's default cost model XGBoost: {score_p1}\n"
                    )
    
                    # --- Grandparent schedule ---
                    grandparent_node = parent_node.parent
                    if grandparent_node and grandparent_node.schedule is not None:
                        p2_sch = grandparent_node.schedule
                        scores_p2 = self._predict_normalized_score([p2_sch])
                        score_p2 = scores_p2[0] if scores_p2 else 0.0
    
                        try:
                            p2_mod_str = p2_sch.mod.script()
                        except Exception:
                            p2_mod_str = "<failed to script IR>"
                        p2_trace_str = str(p2_sch.trace)
                        historical_perf_parts.append(
                            "Grandparent Schedule:\n"
                            f"Grandparent's IR:\n{p2_mod_str}\n\n"
                            f"Grandparent's Trace:\n{p2_trace_str}\n\n"
                            f"Grandparent's Predicted Score by TVM's default cost model XGBoost: {score_p2}\n"
                        )

            except Exception as e:
                if self.verbose >= 1:
                    logger.warning("Failed to gather historical info for Leaf/Parent/Grandparent: %s", str(e))
    
            historical_perf = "\n\n".join(historical_perf_parts) if historical_perf_parts else None
            short_hist_perf = "\n\n".join(short_hist_perf_parts) if short_hist_perf_parts else None
            logger.warning("Invoking LLM policy to pick a sequence of mutators and the next model to be used.")
            possible_mutator_names = [str(m) for m in self._mutator_probs.keys()]
            mutator_probs_dict = {str(mut): prob for mut, prob in self._mutator_probs.items()}
    
            chosen_mutator_names: Optional[List[str]] = None
            next_model_name: Optional[str] = None

            logger.warning(
                "[LLM] Calling pick_mutators: leaf_depth=%d current_model=%s llm_budget=%d",
                leaf.depth, leaf.model_name, self.llm_budget
            )
            model_performance = self._build_model_performance_string(leaf)
            start_t = time.time()
            chosen_mutator_names, next_model_name = self.llm_policy.pick_mutators(
                mod=leaf.schedule.mod,
                available_mutators=possible_mutator_names,
                llm_bucket=self.llm_bucket,
                historical_perf=historical_perf,
                available_mutator_probs=mutator_probs_dict,
                current_model=leaf.model_name,
                model_performance=model_performance,
            )

            elapsed = time.time() - start_t

            if (not next_model_name) or (next_model_name not in self.llm_bucket):
                best_model = self._best_global_model_by_hit_rate()
                if best_model is not None:
                    logger.warning(
                        "[LLM] Invalid/missing next_model=%r. Falling back to best global model by hit_rate: %s",
                        next_model_name, best_model
                    )
                    next_model_name = best_model
                else:
                    # No scored stats yet -> fall back safely
                    fallback = leaf.model_name or self.default_llm_model_name
                    logger.warning(
                        "[LLM] Invalid/missing next_model=%r and no global hit-rate data yet. Falling back to %r",
                        next_model_name, fallback
                    )
                    next_model_name = fallback

            logger.warning(
                "[LLM] pick_mutators returned: leaf_depth=%d current_model=%s elapsed=%.3fs mutators=%s next_model=%s",
                leaf.depth, leaf.model_name, elapsed, chosen_mutator_names, next_model_name
            )
            leaf.model_call_count += 1
            leaf.model_cumulative_latency += elapsed

            cur_model = leaf.model_name or "UNKNOWN_MODEL"
            st = self._llm_model_stats.setdefault(cur_model, ModelStats())
            st.n_calls += 1
            st.total_latency += float(elapsed)

            if (not next_model_name) or (next_model_name not in self.llm_bucket):
                st.n_errors+=1
            if not chosen_mutator_names:
                st.n_errors+=1
            self._log_llm_bucket_call_counts(where=f"after LLM call (used={cur_model})")
            self.llm_budget = max(0, self.llm_budget - 1)
            logger.warning("LLM budget decremented. Remaining: %d", self.llm_budget)

            if not chosen_mutator_names:
                logger.warning(
                    "LLM returned no valid mutators (or call failed). Falling back to random expansion."
                )
                child = _random_expand()
            
                # OPTIONAL: if you want to preserve the LLM's suggested model switch
                if child is not None and next_model_name:
                    child.model_name = next_model_name
            
                return child
            logger.warning("LLM returned mutator names: '%s'", chosen_mutator_names)
            
            temp_sch = leaf.schedule
            success = False
            
            for name in chosen_mutator_names:
                logger.warning("[LLM] Applying mutator '%s' at leaf_depth=%d", name, leaf.depth)
            
                chosen_mutator = None
                for mut, _prob in self._mutator_probs.items():
                    if str(mut) == name:
                        chosen_mutator = mut
                        break
                    
                if chosen_mutator is None:
                    logger.warning(
                        "LLM mutator name '%s' did not match any known mutator. Skipping.",
                        name
                    )
                    continue
                
                maybe_new = self._apply_mutator_with_retry(temp_sch, chosen_mutator, rand_state)
                if maybe_new is None:
                    logger.warning(
                        "Failed applying mutator '%s'. Continuing with remaining mutators.",
                        name
                    )
                    continue
                
                temp_sch = maybe_new
                success = True
                logger.warning("[LLM] Mutator '%s' applied successfully.", name)
            
            new_sch = temp_sch if success else None
            if new_sch is None:
                logger.warning(
                    "LLM mutators produced no valid new schedule. Falling back to random expansion."
                )
                child = _random_expand()
            
                if child is not None and next_model_name:
                    child.model_name = next_model_name
            
                return child
                
            child_score: Optional[float] = None
            try:
                child_score_list = self._predict_normalized_score([new_sch])
                child_score = child_score_list[0] if child_score_list else 0.0

                st_reg = self._llm_model_stats.setdefault(cur_model, ModelStats())
                st_reg.n_scored += 1
                if child_score > leaf_score:
                    st_reg.n_hits += 1
            except Exception:
                child_score = None

            is_small_call = self._is_small_model(cur_model)
            is_regression = False

            if is_small_call and (child_score is not None):
                is_regression = (child_score < leaf_score - 1e-12)

                if is_regression:
                    self._small_model_regression_streak += 1
                else:
                    self._small_model_regression_streak = 0

                logger.warning(
                    "[LLM] small_model_regression_streak=%d (model=%s) leaf_score=%.6f child_score=%.6f",
                    self._small_model_regression_streak,
                    cur_model,
                    float(leaf_score),
                    float(child_score),
                )

            do_alt = (
                is_small_call
                and (child_score is not None)
                and (child_score < leaf_score - 1e-12)
                and (self._small_model_regression_streak > 0)
                and (self._small_model_regression_streak % self._corrector_every_k_regressions == 0)
                and (self.llm_policy is not None)
                and (self._alt_model() is not None)
                and (self.llm_budget > 0)
            )

            if do_alt:
                corrector_model = self._alt_model()  # default/large model
                assert corrector_model is not None

                logger.warning(
                    "[LLM-CORRECTOR] Triggered: streak=%d (every %d). small_model=%s corrector_model=%s",
                    self._small_model_regression_streak,
                    self._corrector_every_k_regressions,
                    cur_model,
                    corrector_model,
                )

                corrector_start = time.time()
                v_mutators, v_next_model = self.llm_policy.corrector_mutators(
                    mod=leaf.schedule.mod,
                    available_mutators=possible_mutator_names,
                    llm_bucket=self.llm_bucket,
                    historical_perf=short_hist_perf,
                    available_mutator_probs=mutator_probs_dict,
                    corrector_model=corrector_model,
                    small_model_name=cur_model,
                    small_mutators=chosen_mutator_names or [],
                    small_next_model=next_model_name,
                    leaf_score=leaf_score,
                    small_child_score=child_score,
                    model_performance=model_performance,
                )
                corrector_elapsed = time.time() - corrector_start
                st_large = self._llm_model_stats.setdefault(corrector_model, ModelStats())
                st_large.n_corrector_calls += 1
                st_large.corrector_total_latency += float(corrector_elapsed)
                self.llm_budget = max(0, self.llm_budget - 1)
                logger.warning("[LLM-CORRECT] Budget decremented (corrector). Remaining=%d", self.llm_budget)
                if (not v_next_model) or (v_next_model not in self.llm_bucket):
                    st_large.n_corrector_errors += 1
                    # fallback: keep already-sanitized next_model_name from small call
                    v_next_model = next_model_name
                if not v_mutators:
                    st_large.n_corrector_errors += 1
                    logger.warning("[LLM-CORRECT] Corrector returned no mutators. Keeping small model schedule.")
                else:
                    corrector_sch = self._apply_mutator_names_sequence(
                        base_sch=leaf.schedule,
                        mutator_names=v_mutators,
                        rand_state=rand_state,
                        log_prefix="[LLM-CORRECT]",
                        leaf_depth=leaf.depth,
                    )
                    if corrector_sch is None:
                        logger.warning(
                            "[LLM-CORRECT] Corrector mutators produced no valid schedule. Keeping small model schedule."
                        )
                    else:
                        v_score: Optional[float] = None
                        try:
                            v_score_list = self._predict_normalized_score([corrector_sch])
                            v_score = v_score_list[0] if v_score_list else 0.0

                            st_large.n_corrector_scored += 1
                            if v_score > leaf_score:
                                st_large.n_corrector_hits += 1
                        except Exception:
                            v_score = None

                        replaced_small_child = False
                        if (child_score is not None) and (v_score is not None):
                            if v_score < child_score - 1e-12:
                                logger.warning(
                                    "[LLM-CORRECT] Corrected schedule worse than small schedule "
                                    "(v=%.6f < small=%.6f). Keeping small child.",
                                    float(v_score),
                                    float(child_score),
                                )
                            else:
                                new_sch = corrector_sch
                                chosen_mutator_names = v_mutators
                                next_model_name = v_next_model
                                replaced_small_child = True
                        else:
                            new_sch = corrector_sch or new_sch
                            if corrector_sch is not None:
                                chosen_mutator_names = v_mutators
                                next_model_name = v_next_model
                                replaced_small_child = True

                        logger.warning(
                            "[LLM-CORRECT] leaf_score=%.6f small_child_score=%s corrector_score=%s replaced_small_child=%s",
                            float(leaf_score),
                            "N/A" if child_score is None else f"{float(child_score):.6f}",
                            "N/A" if v_score is None else f"{float(v_score):.6f}",
                            "YES" if replaced_small_child else "NO",
                        )

                logger.warning(
                    "[LLM-CORRECT] stats: corrector_calls=%d corrector_scored=%d corrector_hits=%d corrector_errors=%d",
                    int(st_large.n_corrector_calls),
                    int(st_large.n_corrector_scored),
                    int(st_large.n_corrector_hits),
                    int(st_large.n_corrector_errors),
                )
                self._log_llm_bucket_call_counts(where=f"after CORRECTOR call (model={corrector_model})")

            child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
            child.model_name = next_model_name

            logger.warning(
                "[LLM] Created child node: parent_depth=%d child_depth=%d child_model=%s",
                leaf.depth, child.depth, child.model_name
            )
            leaf.children.append(child)

            logger.warning(
                "Successfully expanded leaf using %s approach. New child node at depth %d.",
                "LLM-based",
                child.depth
            )
            return child



    def _simulate_node(self, node: MCTSNode, rand_state: int) -> float:
        if (self.mcts_num_rollouts_per_expansion <= 1) and (self.mcts_num_threads <= 1):
            return self._rollout(node.schedule, node.depth, rand_state)

        results = []
        if (self.mcts_num_threads > 1) and (self.mcts_num_rollouts_per_expansion > 1):
            with ThreadPoolExecutor(max_workers=self.mcts_num_threads) as executor:
                futures = [
                    executor.submit(self._rollout, node.schedule, node.depth, rand_state)
                    for _ in range(self.mcts_num_rollouts_per_expansion)
                ]
                for f in as_completed(futures):
                    results.append(f.result())
        else:
            for _ in range(self.mcts_num_rollouts_per_expansion):
                results.append(self._rollout(node.schedule, node.depth, rand_state))

        if results:
            return sum(results) / len(results)
        return 0.0

    def _backprop(self, node: MCTSNode, value: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _rollout(self, schedule: Schedule, depth: int, rand_state: int) -> float:
        rs0 = self._search_state.next_rand_state() if self._search_state else (rand_state or 1)
        new_sch = self._replay_schedule(schedule.trace, rs0)
        if new_sch is None:
            return 0.0

        cur_depth = depth
        while (self.mcts_max_depth is None) or (cur_depth < self.mcts_max_depth):
            cur_depth += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                logger.warning("[_rollout] No mutator found (mut is None). Breaking from rollout loop.")
                break
            rs = self._search_state.next_rand_state() if self._search_state else (rand_state or 1)
            
            mutated_trace = self._mutator_apply_trace(mut, new_sch.trace, rs)
            if mutated_trace is None:
                break
            maybe_new = self._replay_schedule(mutated_trace, rs)
            if maybe_new is None:
                logger.warning("[_rollout] Replaying the mutated trace returned None. Stopping mutations.")
                break
            
            new_sch = maybe_new

        if not self._cost_model:
            logger.warning(
                f"[_rollout] No cost_model found. Returning random fallback score."
            )
            rng = getattr(self._search_state, "_rng", random)
            return rng.random()
        arg_info = ArgInfo.from_entry_func(new_sch.mod, remove_preproc=True)
        candidate = MeasureCandidate(new_sch, arg_info)
        preds = self._cost_model.predict(self._ctx, [candidate])
        if preds:
            return max(0.0, preds[0])
        return 0.0

    def gather_unmeasured_leaves(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Traverse the tree to find leaf nodes (no children) that haven't been measured yet.
        """
        stack = [node]
        leaves = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None and not nd.children:
                wl = None
                wl = self._commit_workload_cached(nd.schedule)
                if wl not in self._measured_workloads:
                    leaves.append(nd)
            else:
                stack.extend(nd.children)
        return leaves

    def pick_unmeasured_best_leaves(self, root: MCTSNode, batch_size: int) -> List[Schedule]:
        leaves = self.gather_unmeasured_leaves(root)
        if not leaves:
            return []
        scored = []
        for nd in leaves:
            q_val = (nd.total_value / nd.visits) if nd.visits > 0 else 0.0
            scored.append((nd, q_val))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored[:batch_size]
        return [node.schedule for (node, _) in top_nodes]

    def _gather_tree_schedules(self, root: MCTSNode) -> List[MCTSNode]:
        stack = [root]
        out_nodes = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None:
                out_nodes.append(nd)
            stack.extend(nd.children)
        return out_nodes


    def _replay_schedule(self, trace: Optional[Trace], rand_state: int) -> Optional[Schedule]:
        """
        Rebuild a Schedule from a trace, ignoring built-in postproc so we can do our own.
        Then apply our postprocs to ensure correctness and constraints.
        """
        if not self._ctx or not self._ctx.mod:
            return None
        mod = self._ctx.mod

        if trace is None:
            try:
                sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            except (InvalidScheduleError, tvm.TVMError):
                return None
            sch.enter_postproc()
            if not self._apply_postprocs(sch):
                return None
            return sch

        try:
            sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None

        sch.enter_postproc()
        if not self._apply_postprocs(sch):
            return None
        return sch

    def _apply_postprocs(self, sch: Schedule) -> bool:
        if not self._postprocs:
            return True

        ffi_postproc = getattr(_ffi_api, "SearchStrategyApplyPostprocs", None)
        if ffi_postproc is not None:
            try:
                return bool(ffi_postproc(sch, self._postprocs))
            except Exception:
                pass

        for proc in self._postprocs:
            try:
                if not proc.apply(sch):
                    print(f"[DEBUG] Postproc '{proc}' rejected the schedule (returned False).", flush=True)
                    return False
            except (InvalidScheduleError, tvm.TVMError) as e:
                print(f"[DEBUG] Postproc '{proc}' CRASHED. Reason: {e}", flush=True)
                return False
        return True


    def _pick_random_mutator(self, rand_state: int) -> Optional[Mutator]:
        if not self._mutator_probs:
            return None
        total_p = sum(self._mutator_probs.values())
        rng = getattr(self._search_state, "_rng", random)
        r = rng.random() * total_p
        s = 0.0
        for mut, p in self._mutator_probs.items():
            s += p
            if r <= s:
                return mut
        # fallback
        return list(self._mutator_probs.keys())[0]
    
    def _mutator_apply_trace(self, mut: Mutator, trace: Trace, rs: int) -> Optional[Trace]:
        try:
            try:
                return mut.apply(trace, rs)
            except TypeError:
                # Older signature: apply(trace)
                return mut.apply(trace)
        except (InvalidScheduleError, tvm.TVMError):
            return None

    def _try_mcts_mutation(self, parent_sch: Schedule, rand_state: int) -> Optional[Schedule]:
        attempts = 0
        while attempts <= self.genetic_max_fail_count:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                rs = self._search_state.next_rand_state() if self._search_state else (rand_state or 1)
                child_sch = self._replay_schedule(parent_sch.trace, rs)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    # wl = self._database.commit_workload(child_sch.mod)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
                continue

           
            rs = self._search_state.next_rand_state() if self._search_state else (rand_state or 1)

            new_trace = self._mutator_apply_trace(mut, parent_sch.trace, rs)

            if new_trace is None:
                self._mutator_failure_count[mut] = self._mutator_failure_count.get(mut, 0) + 1
            else:
                child_sch = self._replay_schedule(new_trace, rs)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
        return None

    def _apply_mutator_with_retry(
        self,
        parent_sch: tvm.tir.Schedule,
        chosen_mutator: Mutator,
        rand_state: int
    ) -> Optional[tvm.tir.Schedule]:
        attempts = 0
        while attempts <= self.genetic_max_fail_count:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            rs = self._search_state.next_rand_state() if self._search_state else (rand_state or 1)
            new_trace = self._mutator_apply_trace(chosen_mutator, parent_sch.trace, rs)

            if new_trace is None:
                print(f"[DEBUG] Mutator {chosen_mutator} failed.", flush=True)
                self._mutator_failure_count[chosen_mutator] = (
                    self._mutator_failure_count.get(chosen_mutator, 0) + 1
                )
            else:
                child_sch = self._replay_schedule(new_trace, rs)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
                    
            print(f"[DEBUG] Replay/PostProc failed for {chosen_mutator}", flush=True)
    
        return None
    
    def _apply_mutator_names_sequence(
        self,
        base_sch: tvm.tir.Schedule,
        mutator_names: List[str],
        rand_state: int,
        log_prefix: str = "[LLM]",
        leaf_depth: Optional[int] = None,
    ) -> Optional[tvm.tir.Schedule]:
        if not mutator_names:
            return None

        temp_sch = base_sch
        success = False

        for name in mutator_names:
            if leaf_depth is not None:
                logger.warning("%s Applying mutator '%s' at leaf_depth=%d", log_prefix, name, leaf_depth)
            else:
                logger.warning("%s Applying mutator '%s'", log_prefix, name)

            chosen_mutator = None
            for mut, _prob in self._mutator_probs.items():
                if str(mut) == name:
                    chosen_mutator = mut
                    break

            if chosen_mutator is None:
                logger.warning("%s Mutator name '%s' did not match any known mutator. Skipping.", log_prefix, name)
                continue

            maybe_new = self._apply_mutator_with_retry(temp_sch, chosen_mutator, rand_state)
            if maybe_new is None:
                logger.warning("%s Failed applying mutator '%s'. Continuing.", log_prefix, name)
                continue

            temp_sch = maybe_new
            success = True
            logger.warning("%s Mutator '%s' applied successfully.", log_prefix, name)

        return temp_sch if success else None



    def _commit_workload_cached(self, sch: Schedule) -> Optional[Workload]:
        if self._database is None:
            return None
        wl = getattr(sch, "_cached_wl", None)
        if wl is not None:
            return wl
        shash = tvm.ir.structural_hash(sch.mod)
        wl = self._workload_cache.get(shash)
        if wl is None:
            wl = self._database.commit_workload(sch.mod)
            self._workload_cache[shash] = wl
        sch._cached_wl = wl
        return wl

    def _predict_normalized_score(self, schedules: List[Schedule]) -> List[float]:
        if not schedules or not self._cost_model:
            return [0.0] * len(schedules)
        cands = []
        for sch in schedules:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            cands.append(MeasureCandidate(sch, arg_info))
        scores = self._cost_model.predict(self._ctx, cands)
        return [max(0.0, sc) for sc in scores]

    @property
    def _measured_workloads(self) -> Set[Workload]:
        """
        The set of workload keys that have been actually measured on hardware.
        """
        if self._search_state is not None:
            return self._search_state.measured_workloads
        return set()

    @property
    def _seen_workloads(self) -> Set[Workload]:
        """
        The set of workload keys we've encountered in generated schedules.
        """
        if self._search_state is not None:
            return self._search_state.seen_workloads
        return set()


class MCTSTuningState:
    """
    MCTSTuningState tracks the MCTS root, population, # of trials used,
    best score, etc. The MCTSTuner performs expansions and rollouts; 
    MCTSTuningState decides how to handle each iteration (e.g. picking 
    unmeasured leaves, ranking population, etc.).
    """

    def __init__(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"],
        cost_model: Optional["CostModel"],
        context: "TuneContext",
        tuner: MCTSTuner,
    ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_spaces = design_spaces
        self.database = database
        self.cost_model = cost_model
        self.context = context
        self.tuner = tuner
        self.tuner.attach_search_state(self)

        self.mod = context.mod
        self.workload_key = None
        if self.database and self.mod is not None:
            self.workload_key = self.database.commit_workload(self.mod)
            # also store in tuner for cost model usage
            self.tuner._workload_key = self.workload_key

        self.trial_count = 0
        self.num_empty_iters = 0
        self.used_init_population = False
        self.population: List[Tuple[Schedule, bool]] = []
        self.mcts_root: Optional[MCTSNode] = None

        self.measured_workloads: Set[Workload] = set()
        self.seen_workloads: Set[Workload] = set()

        self.best_score_so_far = -float("inf")
        self.stale_iter_count = 0
        self.stale_diversity_count = 0
        self.diversity_history: List[float] = []
        self.score_history: List[float] = []
        self.dynamic_pop_size = self.tuner.population_size

        rs = context.rand_state
        self.rand_state = rs if rs is not None else 1
        if self.rand_state == 0:
            self.rand_state = 1
        self._rng = random.Random(self.rand_state)

    def next_rand_state(self) -> int:
        rs = self._rng.getrandbits(31)
        return rs if rs != 0 else 1

    def reset(self) -> None:
        """
        Called from MCTSSearch.post_tuning().
        """
    
    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        if self.tuner.verbose >= 1:
            logger.warning(
                "[DEBUG] Enter generate_measure_candidates: trial_count=%d, max_trials=%d",
                self.trial_count, self.max_trials
            )

        if self.trial_count >= self.max_trials:
            return None

        remaining = self.max_trials - self.trial_count
        batch_size = min(remaining, self.num_trials_per_iter)
        if batch_size <= 0:
            return None

        if not self.used_init_population:
            init_pop = self._init_population()
            if not init_pop:
                return None
            self.mcts_root = MCTSNode(schedule=None, parent=None, depth=0)
            for (sch, is_measured) in init_pop:
                child = MCTSNode(schedule=sch, parent=self.mcts_root, depth=1)
                child.model_name=self.tuner.default_llm_model_name
                self.mcts_root.children.append(child)
            self.population = init_pop
            self.used_init_population = True

            if self.tuner.verbose >= 1:
                logger.warning(
                    "generate_measure_candidates: MCTS: Initialized root with %d child schedules.",
                    len(init_pop)
                )

        self.population = self.tuner.explore(
            mcts_root=self.mcts_root,
            population=self.population,
            dynamic_pop_size=self.dynamic_pop_size,
            rand_state=self.rand_state,
        )

        if not self.population:
            self.num_empty_iters += 1
            logger.warning(
                "generate_measure_candidates: MCTS: explore() returned empty => empty iters=%d",
                self.num_empty_iters
            )
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                if self.tuner.verbose >= 1:
                    logger.warning("generate_measure_candidates: MCTS: Stopping early => repeated empty iters.")
                return None
            return None

        logger.warning(
            "generate_measure_candidates: MCTS: population size=%d before eps-greedy picking",
            len(self.population)
        )

        cands_sch = self._pick_unmeasured_eps_greedy(self.population, batch_size, self.rand_state)
        if not cands_sch:
            self.num_empty_iters += 1
            logger.warning(
                "generate_measure_candidates: MCTS: no unmeasured schedules => empty iters=%d",
                self.num_empty_iters
            )
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                if self.tuner.verbose >= 1:
                    logger.warning("generate_measure_candidates: stopping early => repeated empty iters.")
                return None
            return None

        logger.warning(
            "generate_measure_candidates: [DEBUG] Eps-greedy picked %d schedules for measurement (batch_size=%d).",
            len(cands_sch), batch_size
        )

        measure_cands: List[MeasureCandidate] = []
        for sch in cands_sch:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            measure_cands.append(MeasureCandidate(sch, arg_info))

        logger.warning(
                "generate_measure_candidates: [DEBUG] MCTS => returning %d cands; trial_count=%d, "
                "batch_size_requested=%d, used_init_population=%s",
                len(measure_cands),
                self.trial_count,
                batch_size,
                str(self.used_init_population),
            )
        return measure_cands


    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        if self.database is None:
            logger.warning("database is not defined, skipping MCTS measure update.")
            return

        num_measured_now = 0
        best_run_sec = float("inf")
        for cand, res in zip(measure_candidates, results):
            sch = cand.sch
            mod = sch.mod
            wl = self.database.commit_workload(mod)
            if res.run_secs and all(t >= 0 for t in res.run_secs):
                run_sec = sum(res.run_secs) / len(res.run_secs)
                if run_sec < best_run_sec:
                    best_run_sec = run_sec
                self.measured_workloads.add(wl)
                self._mark_schedule_measured(sch)
                num_measured_now += 1

        self.trial_count += num_measured_now
        if best_run_sec < float("inf"):
            new_score = 1.0 / best_run_sec
            self.score_history.append(new_score)
            if new_score > self.best_score_so_far + 1e-12:
                self.best_score_so_far = new_score
                self.stale_iter_count = 0
            else:
                self.stale_iter_count += 1
                if self.stale_iter_count >= self.tuner.max_stale_iters and self.tuner.verbose >= 1:
                    logger.warning(
                        "notifu_runner_results: MCTS: No improvement => stopping early (stale_iter=%d).",
                        self.stale_iter_count
                    )
        else:
            self.score_history.append(0.0)

        if self.tuner.verbose >= 1:
            logger.warning(
                "MCTS: notify_runner_results => measured=%d, total=%d, stale_iter=%d, div_stale=%d",
                num_measured_now, self.trial_count, self.stale_iter_count, self.stale_diversity_count
            )

    def _init_population(self) -> List[Tuple[Schedule, bool]]:
        num_measured_wanted = int(self.tuner.population_size * self.tuner.init_measured_ratio)
        measured_from_db = self._pick_best_from_database(num_measured_wanted)

        need_rand = max(
            self.tuner.population_size - len(measured_from_db),
            self.tuner.init_min_unmeasured
        )
        unmeasured_rand = self._sample_init_population(need_rand)

        logger.warning(
            "[MCTS init_pop] from DB: %d, from random: %d, population_size=%d, init_min_unmeasured=%d",
            len(measured_from_db),
            len(unmeasured_rand),
            self.tuner.population_size,
            self.tuner.init_min_unmeasured
        )

        combined = [(sch, True) for sch in measured_from_db] + \
                   [(sch, False) for sch in unmeasured_rand]

        if len(combined) < self.tuner.init_min_unmeasured and self.tuner.verbose >= 1:
            logger.warning("MCTS: Could not collect enough unmeasured schedules.")

        self._rng.shuffle(combined)
        if len(combined) > self.tuner.population_size:
            combined = combined[: self.tuner.population_size]

        for (sch, measured_flag) in combined:
            wl = self.tuner._commit_workload_cached(sch)
            # wl = self.database.commit_workload(sch.mod)
            self.seen_workloads.add(wl)
            if measured_flag:
                self.measured_workloads.add(wl)
        return combined

    def _pick_best_from_database(self, num: int) -> List[Schedule]:
        if num <= 0 or not self.database:
            return []
        out = []
        top_records = self.database.get_top_k(self.workload_key, num)
        for rec in top_records:
            seed = self.next_rand_state()
            sch = self._replay_schedule(rec.trace, seed=seed)
            if sch is not None:
                wl = self.tuner._commit_workload_cached(sch)
                # wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
        return out

    def _replay_schedule(self, trace: Optional[Trace], seed: Optional[int] = None) -> Optional[Schedule]:
        if self.context is None or self.context.mod is None:
            return None
        if trace is None:
            return None

        mod = self.context.mod
        s = seed if (seed is not None and seed != 0) else self.next_rand_state()

        try:
            sch = Schedule(mod, seed=s, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None

        sch.enter_postproc()

        for proc in self.tuner._postprocs:
            try:
                if not proc.apply(sch):
                    return None
            except (InvalidScheduleError, tvm.TVMError):
                return None

        return sch

    def _sample_init_population(self, num: int) -> List[Schedule]:
        if not hasattr(self, "_logged_mutators"):
            print(f"[DEBUG] Available Mutators & Probs: {self.tuner._mutator_probs}", flush=True)
            self._logged_mutators = True
        out = []
        fails = 0
        n_spaces = len(self.design_spaces)
        while len(out) < num and fails < self.tuner.max_fail_count:
            idx = self._rng.randrange(n_spaces)
            base_sch = self.design_spaces[idx]
            seed = self.next_rand_state()
            sch = self._replay_schedule(base_sch.trace, seed=seed)
            if sch is not None:
                wl = self.tuner._commit_workload_cached(sch)
                # wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
                    self.seen_workloads.add(wl)
                else:
                    fails += 1
            else:
                fails += 1
        return out
    
    def _pick_unmeasured_eps_greedy(
        self,
        schedules_with_flags: List[Tuple[tvm.tir.Schedule, bool]],
        total_needed: int,
        rand_state: int
    ) -> List[tvm.tir.Schedule]:
        logger.warning(
            "[DEBUG] _pick_unmeasured_eps_greedy called with total_needed=%d, eps_greedy=%.3f",
            total_needed, 0.05
        )
        unmeasured = []
        for (sch, measured_flag) in schedules_with_flags:
            if not measured_flag:
                unmeasured.append(sch)

        logger.warning("[DEBUG] Found %d unmeasured schedules.", len(unmeasured))

        if not unmeasured:
            return []
        preds = self.tuner._predict_normalized_score(unmeasured)
        logger.warning("[DEBUG] Computed cost-model predictions for %d unmeasured schedules.", len(preds))

        scored = list(zip(unmeasured, preds))
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.warning(
            "[DEBUG] Top schedule after sorting has predicted score=%.4f if the list is non-empty.",
            scored[0][1] if scored else -1.0
        )

        n_total = min(total_needed, len(scored))
        n_rand = int(round(n_total * 0.05))
        n_top = n_total - n_rand

        logger.warning(
            "[DEBUG] Eps-greedy selection: total_needed=%d => n_top=%d, n_rand=%d",
            n_total, n_top, n_rand
        )

        top_part = scored[:n_top]
        leftover = scored[n_top:]

        random_schedules = []
        if leftover and n_rand > 0:
            n_rand = min(n_rand, len(leftover))

            rng = self._rng
            random_part = rng.sample(leftover, n_rand)
            random_schedules = [sch for (sch, _) in random_part]

        top_schedules = [sch for (sch, _) in top_part]

        combined = top_schedules + random_schedules
        logger.warning(
            "[DEBUG] _pick_unmeasured_eps_greedy => returning %d schedules => %d top + %d random",
            len(combined), len(top_schedules), len(random_schedules)
        )

        return combined


    def _mark_schedule_measured(self, sch: Schedule):
        wl = self.database.commit_workload(sch.mod)
        self.measured_workloads.add(wl)
        for i, (pop_sch, was_measured) in enumerate(self.population):
            if pop_sch == sch and not was_measured:
                self.population[i] = (pop_sch, True)

    def _predict_population_scores(self, pop: List[Tuple[Schedule, bool]]) -> List[float]:
        schs = [p[0] for p in pop]
        if not schs:
            return []
        return self.tuner._predict_normalized_score(schs)

    def _check_population_diversity(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        mean_val = sum(scores) / len(scores)
        var = sum((s - mean_val) ** 2 for s in scores) / len(scores)
        cur_div = math.sqrt(var)
        self.diversity_history.append(cur_div)
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        avg_div = sum(self.diversity_history) / len(self.diversity_history)
        self.tuner.diversity_epsilon = 0.5 * avg_div
        return cur_div

@derived_object
class MCTSSearchPyFull(PySearchStrategy):

    def __init__(
        self,
        population_size: int,
        init_measured_ratio: float,
        init_min_unmeasured: int,
        max_fail_count: int,
        genetic_num_iters: int,
        genetic_mutate_prob: float,
        genetic_max_fail_count: int,
        num_empty_iters_before_early_stop: int = 100,
        max_stale_iters: int = 60,
        diversity_epsilon: float = 1e-6,
        max_stale_diversity_iters: int = 30,
        trace_commit: bool = True,
        verbose: int = 2,
        # MCTS-specific:
        mcts_ucb_constant: float = 1.41,
        mcts_max_depth: Optional[int] = 500,
        mcts_num_threads: int = 1,
        mcts_num_rollouts_per_expansion: int = 1,
        use_llm: bool = False,
        llm_budget: int = 1,
        default_llm_model_name: str = "",
        llm_bucket: Optional[List[str]] = None,
        mcts_model_balance_lambda: float = 0.0,
        llm_param_count: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob
        self.genetic_max_fail_count = genetic_max_fail_count
        self.num_empty_iters_before_early_stop = num_empty_iters_before_early_stop
        self.max_stale_iters = max_stale_iters
        self.diversity_epsilon = diversity_epsilon
        self.max_stale_diversity_iters = max_stale_diversity_iters
        self.trace_commit = trace_commit
        self.verbose = verbose

        self.mcts_ucb_constant = mcts_ucb_constant
        self.mcts_max_depth = mcts_max_depth
        self.mcts_num_threads = mcts_num_threads
        self.mcts_num_rollouts_per_expansion = mcts_num_rollouts_per_expansion

        self.use_llm = use_llm
        self.llm_budget = llm_budget

        self._ctx: Optional["TuneContext"] = None
        self._postprocs: List[Postproc] = []
        self._mutator_probs: Dict[Mutator, float] = {}
        self.state: Optional[MCTSTuningState] = None

        self.default_llm_model_name = default_llm_model_name
        if llm_bucket is not None:
            self.llm_bucket = list(llm_bucket)
        else:
            if self.use_llm:
                self.llm_bucket = [
                    "gpt-5.2",
                    "gpt-5.2-mini",
                ]
                if not self.default_llm_model_name:
                    self.default_llm_model_name = self.llm_bucket[0]
            else:
                self.llm_bucket = []
        self.mcts_model_balance_lambda = float(mcts_model_balance_lambda)
        self.llm_param_count: Dict[str, float] = dict(llm_param_count or {})


    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        self._ctx = context
        if context.space_generator is None:
            raise ValueError("TuneContext.space_generator must be defined.")
        if context.target is None:
            raise ValueError("TuneContext.target must be defined.")

        sg = context.space_generator
        self._postprocs = list(sg.postprocs) if sg.postprocs else []

        user_probs = sg.mutator_probs or {}
        for mut, prob_f in user_probs.items():
            p = float(prob_f.value)
            if p > 1e-12:
                self._mutator_probs[mut] = self._mutator_probs.get(mut, 0.0) + p

        target_kind = str(context.target.kind.name)
        if not self._mutator_probs:
            try:
                default_muts = Mutator.create(target_kind)
            except:
                default_muts = Mutator.create("llvm")
            if isinstance(default_muts, dict):
                for m, p2 in default_muts.items():
                    self._mutator_probs[m] = float(p2)
            elif isinstance(default_muts, list) and len(default_muts) > 0:
                p2 = 1.0 / len(default_muts)
                for m in default_muts:
                    self._mutator_probs[m] = p2

        total_p = sum(self._mutator_probs.values())
        if total_p > 1e-12:
            for k in self._mutator_probs:
                self._mutator_probs[k] /= total_p

        if self.verbose >= 1:
            logger.warning(
                "_initialize_with_tune_context: MCTSSearch: Using target=%s, found #mutators=%d, rand_state=%s",
                target_kind, len(self._mutator_probs), str(context.rand_state)
            )

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"],
        cost_model: Optional["CostModel"],
    ) -> None:
        if self.state is not None:
            raise ValueError("MCTSSearch.pre_tuning called without post_tuning after previous run")
        
        if self.use_llm and (not self.default_llm_model_name):
            raise ValueError(
                "default_llm_model_name must be set when use_llm=True"
            )
        if self.use_llm and (self.default_llm_model_name not in self.llm_bucket):
            raise ValueError(
                f"default_llm_model_name={self.default_llm_model_name!r} "
                "must be one of llm_bucket when use_llm=True"
            )

        # build MCTSTuner
        tuner = MCTSTuner(
            population_size=self.population_size,
            init_measured_ratio=self.init_measured_ratio,
            init_min_unmeasured=self.init_min_unmeasured,
            max_fail_count=self.max_fail_count,
            genetic_num_iters=self.genetic_num_iters,
            genetic_mutate_prob=self.genetic_mutate_prob,
            genetic_max_fail_count=self.genetic_max_fail_count,
            num_empty_iters_before_early_stop=self.num_empty_iters_before_early_stop,
            max_stale_iters=self.max_stale_iters,
            diversity_epsilon=self.diversity_epsilon,
            max_stale_diversity_iters=self.max_stale_diversity_iters,
            trace_commit=self.trace_commit,
            verbose=self.verbose,
            mcts_ucb_constant=self.mcts_ucb_constant,
            mcts_max_depth=self.mcts_max_depth,
            mcts_num_threads=self.mcts_num_threads,
            mcts_num_rollouts_per_expansion=self.mcts_num_rollouts_per_expansion,
            postprocs=self._postprocs,
            mutator_probs=self._mutator_probs,
            context=self._ctx,
            cost_model=cost_model,
            database=database,
            workload_key=None,
            use_llm=self.use_llm,
            llm_budget=self.llm_budget,
            llm_policy=LLMGuidancePolicy(
            verbose=True),
            llm_bucket=self.llm_bucket,
            default_llm_model_name=self.default_llm_model_name,
            mcts_model_balance_lambda=self.mcts_model_balance_lambda,
            llm_param_count=self.llm_param_count,
        )

        # build MCTSTuningState
        self.state = MCTSTuningState(
            max_trials=max_trials,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=design_spaces,
            database=database,
            cost_model=cost_model,
            context=self._ctx,
            tuner=tuner,
        )

        if self.verbose >= 1:
            logger.warning(
                "MCTSSearch.pre_tuning => max_trials=%d, num_per_iter=%d, #design_spaces=%d",
                max_trials, num_trials_per_iter, len(design_spaces)
            )

    def post_tuning(self) -> None:
        """
        Called after all tuning is finished. We clear state references, 
        optionally log the best result, etc.
        """
        if self.state:
            self.state.reset()
            self.state = None
        if self.verbose >= 1:
            logger.warning("MCTSSearch: Tuning finished in post_tuning().")

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """
        Called by the MetaSchedule engine each round to get new schedules for measurement.
        """
        if not self.state:
            logger.warning("MCTSSearch.generate_measure_candidates called before pre_tuning.")
            return None
        return self.state.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        if self.state:
            self.state.notify_runner_results(measure_candidates, results)

    def clone(self) -> "MCTSSearchPyFull":
        return MCTSSearchPyFull(
            population_size=self.population_size,
            init_measured_ratio=self.init_measured_ratio,
            init_min_unmeasured=self.init_min_unmeasured,
            max_fail_count=self.max_fail_count,
            genetic_num_iters=self.genetic_num_iters,
            genetic_mutate_prob=self.genetic_mutate_prob,
            genetic_max_fail_count=self.genetic_max_fail_count,
            num_empty_iters_before_early_stop=self.num_empty_iters_before_early_stop,
            max_stale_iters=self.max_stale_iters,
            diversity_epsilon=self.diversity_epsilon,
            max_stale_diversity_iters=self.max_stale_diversity_iters,
            trace_commit=self.trace_commit,
            verbose=self.verbose,
            mcts_ucb_constant=self.mcts_ucb_constant,
            mcts_max_depth=self.mcts_max_depth,
            mcts_num_threads=self.mcts_num_threads,
            mcts_num_rollouts_per_expansion=self.mcts_num_rollouts_per_expansion,
            use_llm=self.use_llm,
            llm_budget=self.llm_budget,
            default_llm_model_name=self.default_llm_model_name,
            llm_bucket=self.llm_bucket,
            mcts_model_balance_lambda=self.mcts_model_balance_lambda,
            llm_param_count=self.llm_param_count,
        )


