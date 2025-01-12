# Lats-Agent-RecSys using LlamaIndex

This repository demonstrates a practical implementation of ``Language Agent Tree Search (LATS)`` using ``LlamaIndex`` in building an intelligent product recommendation system. Built with ``Streamlit`` and powered by the ``SambaNova LLM``, the application provides personalized product recommendations across multiple categories including Cameras, Laptops, Smartphones, and Smart Home Devices. The system integrates real-time market data through ``DuckDuckGo search``, allowing for up-to-date recommendations based on user preferences, budget constraints, and specific feature requirements. 

### Language Agent Tree Search (LATS)

[![Paper](https://img.shields.io/badge/paper-arxiv%202023-red)](https://arxiv.org/pdf/2310.04406)

This repository contains the official implementation of "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" (ICML 2024).

## Overview

LATS is a framework that synergizes the capabilities of Language Models (LMs) in reasoning, acting, and planning. It adapts Monte Carlo Tree Search (MCTS) to language agents to enable more deliberate and adaptive problem-solving compared to existing prompting methods.

Key features:
- Integrates reasoning, acting and planning in a unified framework
- Leverages MCTS with LM-powered value functions for exploration
- Incorporates external feedback and self-reflection
- State-of-the-art results on programming, QA, web navigation, and math tasks

## Core Components

LATS consists of the following key components:

### LM Agent
- Supports both sequential reasoning and decision-making tasks 
- Action space includes both environment actions and reasoning traces
- Samples multiple candidate actions at each step

### Monte Carlo Tree Search
- Adapts MCTS for language agents without requiring environment models
- Uses LM-powered value functions and self-reflections
- Enables planning through state exploration and backpropagation

### Value Function
- Combines LM-generated scores with self-consistency metrics
- λ parameter balances between components (default: 0.5-0.8)
- Guides search towards promising areas of the tree

### Memory & Reflection
- Stores trajectory history and self-reflections
- Enables learning from trial and error
- Improves decision-making through experience

## Usage

### Basic Example

```python
from lats import LATSAgentWorker, AgentRunner
from llama_index.llms.sambanovasystems import SambaNovaCloud

# Initialize the LM
llm = SambaNovaCloud(
    model="Meta-Llama-3.1-70B-Instruct",
    context_window=100000,
    max_tokens=2048,
    temperature=0.1,
)

# Setup the LATS agent
agent_worker = LATSAgentWorker(
    tools=[search_tool],  # Your defined tools
    num_expansions=2,
    max_rollouts=2,
)
agent = AgentRunner(agent_worker)

# Run inference
response = agent.chat(query)
```

### Configuration

Key hyperparameters:
- `num_expansions (n)`: Number of children nodes to expand (default: 5)
- `max_rollouts (k)`: Maximum number of trajectories to sample (default: 50)

## Supported Environments

LATS has been tested on the following environments:

### HotpotQA
- Multi-hop question answering
- Uses Wikipedia web API for information retrieval
- Sample usage in `hotpotqa_example.py`

### Programming (HumanEval/MBPP)
- Program synthesis from natural language
- Uses test suites for external feedback
- Sample usage in `programming_example.py`

### WebShop
- Interactive web navigation task
- Simulated e-commerce environment
- Sample usage in `webshop_example.py`

### Game of 24
- Mathematical reasoning challenge
- Pure internal reasoning task
- Sample usage in `game24_example.py`

## Results

Performance comparisons on key benchmarks:

| Task | Model | Metric | LATS Score | Previous SOTA |
|------|--------|---------|------------|---------------|
| HumanEval | GPT-4 | Pass@1 | 92.7 | 91.0 |
| HotpotQA | GPT-3.5 | EM | 0.71 | 0.54 |
| WebShop | GPT-3.5 | Score | 75.9 | 67.5 |
| Game of 24 | GPT-3.5 | Success Rate | 0.44 | 0.40 |

## Citation

If you use LATS in your research, please cite:

```bibtex
@inproceedings{zhou2024language,
  title={Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models},
  author={Zhou, Andy and Yan, Kai and Shlapentokh-Rothman, Michal and Wang, Haohan and Wang, Yu-Xiong},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
