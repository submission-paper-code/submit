<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# COLT: Lightweight Multi-LLM Collaboration through Shared MCTS Reasoning for Model Compilation
Detailed implementations of this project are included in the folder python/tvm/meta_schedule/search_strategy. This project uses TVM, an open source compiler stack for deep learning systems with [Apache-2.0](LICENSE) license.

Usage:
1. Install TVM and configure the environment as detailed in TVM's documentation.
2. Configure the strategy. Initialize the strategy with hyperparameters.

```
my_strategy = MCTSSearchPyFull(
    population_size=3,
    init_measured_ratio=0,
    init_min_unmeasured=3,
    max_fail_count=20,
    genetic_num_iters=3,
    genetic_mutate_prob=0.85,
    genetic_max_fail_count=2,
    trace_commit=True,
    mcts_num_threads=1,
    mcts_num_rollouts_per_expansion=1,
    use_llm=True,
    llm_budget=500,
)
```
3. Run tuning. Pass the strategy object to tune_tir as a parameter

```
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
    strategy=my_strategy,
)
```
