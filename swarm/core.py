# Standard library imports
import copy
import datetime
import json
from collections import defaultdict
import os
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI

# Azure
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from swarm.convert_azure_response_to_swarm_shape import convert_azure_response_to_swarm_shape

# Local imports
from .util import function_to_json, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result
)

__CTX_VARS_NAME__ = "context_variables"

class Swarm:
    def __init__(self, client=None, ending_tool_names=[], use_azure=False):
        if not client:
            client = OpenAI()
        self.client = client

        self.ending_tool_names = ending_tool_names
        
        self.use_azure = use_azure
        self.azureClient = ChatCompletionsClient(
            endpoint=os.getenv("AZURE_AI_COMPLETION_ENDPOINT", "https://placester-openai.openai.azure.com/openai/deployments/gpt-4o"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_COMPLETION_KEY", "5m1BzeN6qMj8ndXr0NDfDxR2DmxtRWd3qw8c48ghBCv3CjQIShjFJQQJ99BBACYeBjFXJ3w3AAABACOGkzT6")),
            api_version="2024-06-01"
        )

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        model_config = model_override or agent.model
        create_params = {
            "model": model_config,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
            "temperature": agent.temperature,
            "top_p": agent.top_p
        }

        # Only add parallel_tool_calls when not using "o3-mini"
        if tools and not model_config.startswith("o3-mini"):
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        if model_config.startswith("o3-mini"):
            create_params["temperature"] = 1
            create_params["reasoning_effort"] = agent.reasoning_effort or "medium"

        if self.use_azure:
            match create_params["model"]:
                case "gpt-4o":
                    azure_params = create_params
                    azure_params.pop("parallel_tool_calls")
                    return convert_azure_response_to_swarm_shape(self.azureClient.complete(**azure_params))
                case _:
                    return self.client.chat.completions.create(**create_params)
        else:
            return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            
            try:
                args = json.loads(tool_call.function.arguments)

                func = function_map[name]
                # pass context_variables to agent functions
                if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                    args[__CTX_VARS_NAME__] = context_variables
                
                # This is where the function is called - wrap in try/except
                raw_result = function_map[name](**args)
                
                result: Result = self.handle_function_result(raw_result, debug)
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": result.value,
                    }
                )
                partial_response.context_variables.update(result.context_variables)
                if result.agent:
                    partial_response.agent = result.agent
            
            except TypeError as e:
                # Catch argument mismatches and parameter errors
                error_message = f"Error calling function '{name}': {str(e)}. Please check the arguments and try again."
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": error_message,
                    }
                )
            
            except Exception as e:
                # Catch any other exceptions during function execution
                error_message = f"Error executing function '{name}': {str(e)}"
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": error_message,
                    }
                )

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

            for tc in message.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                if name in self.ending_tool_names:
                    break

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)
        token_history: list[dict] = []

        while len(history) - init_len < max_turns and active_agent:
            # get completion with current history, agent

            start_time = datetime.datetime.now()

            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )

            end_time = datetime.datetime.now()

            token_details = {
                "total_tokens": completion.usage.total_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "cached_tokens": completion.usage.prompt_tokens_details.cached_tokens,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
            }

            token_history.append(token_details)

            message = completion.choices[0].message
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

            # Now lets do the break if tool name is in the ending_tool_names
            if partial_response and partial_response.messages:
                ending_tool_called = False  # Flag to indicate if an ending tool was called
                for tool_call_message in partial_response.messages:
                    if isinstance(tool_call_message, dict) and "tool_name" in tool_call_message:
                        if tool_call_message["tool_name"] in self.ending_tool_names:
                            ending_tool_called = True
                            break;
                if ending_tool_called:
                    break

        k = 0
        # Start from init_len to only process new messages
        # (Previous messages are from a different context and shouldn't get token info)
        for i in range(init_len, len(history)):
            if history[i]['role'] == 'assistant':
                if k < len(token_history):
                    history[i].update(token_history[k])
                    k += 1

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
