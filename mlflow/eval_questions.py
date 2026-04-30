"""Seed evaluation questions for the arXiv research agent.

Each record has:
- inputs: the question passed to the agent
- expectations (ground truth):
    * expected_arxiv_ids: canonical paper IDs that a good answer should cite
    * expected_facts: short factual claims the answer should contain

Rules that apply to *every* row (e.g. "must cite with [arxiv:ID]") don't belong
here — those are registered as a `Guidelines` scorer in the notebook.

The IDs below are real arXiv IDs for well-known papers. Feel free to extend.
"""

EVAL_QUESTIONS = [
    {
        "inputs": {
            "question": "What paper introduced the Transformer architecture?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "1706.03762"
            ],
            "expected_facts": [
                "Attention Is All You Need",
                "self-attention"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper introduced BERT and what is its key contribution?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "1810.04805"
            ],
            "expected_facts": [
                "bidirectional",
                "masked language model"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper proposed Low-Rank Adaptation (LoRA) for LLM fine-tuning?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "2106.09685"
            ],
            "expected_facts": [
                "low-rank",
                "fine-tuning"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper introduced Retrieval-Augmented Generation?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "2005.11401"
            ],
            "expected_facts": [
                "retrieval",
                "knowledge-intensive"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper introduced Chain-of-Thought prompting for LLMs?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "2201.11903"
            ],
            "expected_facts": [
                "reasoning",
                "intermediate steps"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper introduced Direct Preference Optimization (DPO)?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "2305.18290"
            ],
            "expected_facts": [
                "preference",
                "reward model"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper introduced the Mixture of Experts approach that Mixtral is based on?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "2401.04088"
            ],
            "expected_facts": [
                "Mixtral",
                "sparse"
            ]
        }
    },
    {
        "inputs": {
            "question": "What paper introduced the ReAct prompting pattern for tool-using agents?"
        },
        "expectations": {
            "expected_arxiv_ids": [
                "2210.03629"
            ],
            "expected_facts": [
                "reasoning",
                "acting"
            ]
        }
    }
]
