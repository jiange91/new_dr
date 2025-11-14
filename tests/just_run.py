from dotenv import load_dotenv
import os
load_dotenv(".env")

import json
from langsmith import Client
import asyncio
from open_deep_research.deep_researcher import deep_researcher_builder
from langgraph.checkpoint.memory import MemorySaver
from tests.evaluators import eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness
import uuid

client = Client()

# NOTE: Configure the right dataset and evaluators
dataset_name = "Deep Research Bench"
exp_name = f"ODR GPT-5, Gensee Search"

# evaluators = [eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness]
evaluators = [eval_overall_quality]
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "gensee" # NOTE: Change to "gensee" for testing
max_researcher_iterations = 6
max_react_tool_calls = 12
summarization_model = "openai:gpt-4.1-mini"
summarization_model_max_tokens = 8192
research_model = "openai:gpt-5" # "anthropic:claude-sonnet-4-20250514"
research_model_max_tokens = 10000
compression_model = "openai:gpt-4.1"
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-4.1"
final_report_model_max_tokens = 10000

async def target(
    inputs: dict,
):
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = max_structured_output_retries
    config["configurable"]["allow_clarification"] = allow_clarification
    config["configurable"]["max_concurrent_research_units"] = max_concurrent_research_units
    config["configurable"]["search_api"] = search_api
    config["configurable"]["max_researcher_iterations"] = max_researcher_iterations
    config["configurable"]["max_react_tool_calls"] = max_react_tool_calls
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_max_tokens"] = summarization_model_max_tokens
    config["configurable"]["research_model"] = research_model
    config["configurable"]["research_model_max_tokens"] = research_model_max_tokens
    config["configurable"]["compression_model"] = compression_model
    config["configurable"]["compression_model_max_tokens"] = compression_model_max_tokens
    config["configurable"]["final_report_model"] = final_report_model
    config["configurable"]["final_report_model_max_tokens"] = final_report_model_max_tokens
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state

example_input = {
    "messages": [
        {
            "type": "human",
            "content": "在当前中国房地产市场低迷的情况下，政府税收减少，这会多大程度上影响地方政府的财政收入",
            "example": False,
            "additional_kwargs": {},
            "response_metadata": {},
        }
    ]
}

if __name__ == "__main__":
    results = asyncio.run(target(example_input))
    print(results)
