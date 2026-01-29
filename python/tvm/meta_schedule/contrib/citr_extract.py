# SPDX‑License‑Identifier: Apache‑2.0
"""CITR extractor – with adaptive‑depth loop lists and “truncated‑reduction” flag."""

import json
from typing import Dict, List, Set
import tvm
from tvm import tir

# ------------------------------------------------------------------ #
# Reduction‑axis kinds across TVM versions                            #
# ------------------------------------------------------------------ #
REDUCE_KINDS = {
    getattr(tir.IterVar, "CommReduce", None),   # 0.12+
    getattr(tir.IterVar, "Reduce", None),       # legacy
}
REDUCE_KINDS = {k for k in REDUCE_KINDS if k is not None}

# ------------------------------------------------------------------ #
# Short dtype tokens                                                  #
# ------------------------------------------------------------------ #
_DTYPE_MAP = {
    "float64": "f64", "float32": "f32", "float16": "f16", "bfloat16": "bf16",
    "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8",
    "uint64": "u64", "uint32": "u32", "uint16": "u16", "uint8": "u8",
}

# ------------------------------------------------------------------ #
# Extractor                                                          #
# ------------------------------------------------------------------ #
class _Extractor:
    MAX_LOOPS_PER_BLOCK = 5

    def __init__(self, func: tir.PrimFunc):
        self.func = func
        self._tensor_id: Dict[str, int] = {}
        self._loop_id: Dict[tir.Var, int] = {}
        self._reduction_loops: Set[int] = set()

        self.tensors: List[Dict] = []
        self.loops: List[Dict] = []
        self.blocks: List[Dict] = []
        self._loop_stack: List[int] = []

        # Pre‑assign IDs to parameter buffers
        for p in func.params:
            self._assign_tensor_id(func.buffer_map[p])

    # ---------- tensor helpers ----------
    @staticmethod
    def _const(expr) -> int:
        return int(expr.value) if isinstance(expr, tir.IntImm) else 0

    def _assign_tensor_id(self, buf: tir.Buffer) -> int:
        if buf.name not in self._tensor_id:
            tid = len(self._tensor_id)
            shape = "x".join(str(self._const(s)) for s in buf.shape)
            raw_scope = getattr(buf, "scope", "global")
            scope = raw_scope() if callable(raw_scope) else raw_scope or "global"
            dtype = _DTYPE_MAP.get(str(buf.dtype), str(buf.dtype))
            self.tensors.append(
                {"id": tid, "name": buf.name, "shape": shape,
                 "dtype": dtype, "scope": scope}
            )
            self._tensor_id[buf.name] = tid
        return self._tensor_id[buf.name]

    # ---------- loop helpers ----------
    def _assign_loop_id(self, loop: tir.For) -> int:
        if loop.loop_var not in self._loop_id:
            lid = len(self._loop_id)
            flag = {
                tir.ForKind.PARALLEL: "p",
                tir.ForKind.VECTORIZED: "v",
                tir.ForKind.UNROLLED: "u",
            }.get(loop.kind, "")
            entry: Dict = {"id": lid, "extent": self._const(loop.extent)}
            if flag:
                entry["flags"] = flag
            self.loops.append(entry)
            self._loop_id[loop.loop_var] = lid
        return self._loop_id[loop.loop_var]

    # ---------- DFS traversal ----------
    def _dfs(self, stmt: tir.Stmt):
        if isinstance(stmt, tir.For):
            lid = self._assign_loop_id(stmt)
            self._loop_stack.append(lid)
            self._dfs(stmt.body)
            self._loop_stack.pop()

        elif isinstance(stmt, tir.BlockRealize):
            self._dfs(stmt.block)

        elif isinstance(stmt, tir.Block):
            if stmt.name_hint == "root":          # skip top‑level
                self._dfs(stmt.body)
                return

            # Mark reductions
            for iv in stmt.iter_vars:
                if iv.iter_type in REDUCE_KINDS and iv.var in self._loop_id:
                    self._reduction_loops.add(self._loop_id[iv.var])

            reads = [self._assign_tensor_id(r.buffer) for r in stmt.reads]
            writes = [self._assign_tensor_id(w.buffer) for w in stmt.writes]

            # ---------- NEW: adaptive depth + reduction propagation ----------
            full_path = list(self._loop_stack)
            if len(full_path) <= self.MAX_LOOPS_PER_BLOCK:
                loops_for_block = full_path                       # keep all
            else:
                loops_for_block = full_path[: self.MAX_LOOPS_PER_BLOCK]
                # If a truncated loop carried a reduction, mark the last kept one
                truncated_part = full_path[self.MAX_LOOPS_PER_BLOCK:]
                if any(lid in self._reduction_loops for lid in truncated_part):
                    last = loops_for_block[-1]
                    self._reduction_loops.add(last)               # flag later
            # ------------------------------------------------------------------

            self.blocks.append(
                {"id": len(self.blocks), "op": stmt.name_hint,
                 "reads": reads, "writes": writes, "loops": loops_for_block}
            )

            if isinstance(stmt.body, tir.Stmt):
                self._dfs(stmt.body)

        elif isinstance(stmt, tir.SeqStmt):
            for sub in stmt.seq:
                self._dfs(sub)

    # ---------- main entry ----------
    def extract(self):
        self._dfs(self.func.body)

        # attach 'r' flag
        for lid in self._reduction_loops:
            entry = self.loops[lid]
            entry["flags"] = entry.get("flags", "") + "r"

        # Compact loop table to used IDs
        used: Set[int] = set()
        for blk in self.blocks:
            used.update(blk["loops"])

        id_map, new_loops = {}, []
        for old_id in self._loop_id.values():
            if old_id not in used:
                continue
            new_id = len(new_loops)
            id_map[old_id] = new_id
            le = self.loops[old_id]
            if le.get("flags", "") == "":
                le = {k: v for k, v in le.items() if k != "flags"}
            new_loops.append({**le, "id": new_id})

        for blk in self.blocks:
            blk["loops"] = [id_map[x] for x in blk["loops"]]

        return {"ir": {"t": self.tensors, "b": self.blocks, "l": new_loops}}


# ---------- public helper ----------
def citr_from_module(mod: tvm.IRModule) -> str:
    """Return CITR JSON string for IRModule `mod`."""
    return json.dumps(_Extractor(mod["main"]).extract(), separators=(",", ":"))
