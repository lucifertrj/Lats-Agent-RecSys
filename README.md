# Lats-Agent-RecSys using LlamaIndex

### Step-by-Step tutorial
[![lats](https://img.youtube.com/vi/22NIh1LZvEY/0.jpg)](https://www.youtube.com/watch?v=22NIh1LZvEY)

This repository demonstrates a practical implementation of ``Language Agent Tree Search (LATS)`` using ``LlamaIndex`` in building an intelligent product recommendation system. Built with ``Streamlit`` and powered by the ``SambaNova LLM``, the application provides personalized product recommendations across multiple categories including Cameras, Laptops, Smartphones, and Smart Home Devices. The system integrates real-time market data through ``DuckDuckGo search``, allowing for up-to-date recommendations based on user preferences, budget constraints, and specific feature requirements. 

### Language Agent Tree Search (LATS)

[![Paper](https://img.shields.io/badge/paper-arxiv%202023-red)](https://arxiv.org/pdf/2310.04406)

This repository contains the official implementation of "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" 

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
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
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
            tools=[search_tool],
            llm=llm
            num_expansions=2,
            max_rollouts=2,
            verbose=True) 
agent = AgentRunner(agent_worker)

# Run inference
response = agent.chat(query)
```

### Configuration

Key hyperparameters:
- `num_expansions (n)`: Number of children nodes to expand (default: 5)
- `max_rollouts (k)`: Maximum number of trajectories to sample (default: 50)

[![lats](https://img.youtube.com/vi/22NIh1LZvEY/0.jpg)](https://www.youtube.com/watch?v=22NIh1LZvEY)
⭐️ LIKE ⭐️ SUBSCRIBE ⭐️ SHARE ⭐️ 

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
