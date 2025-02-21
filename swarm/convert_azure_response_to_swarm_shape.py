# {
#   "choices": [
#     {
#       "content_filter_results": {},
#       "finish_reason": "tool_calls",
#       "index": 0,
#       "logprobs": null,
#       "message": {
#         "content": null,
#         "refusal": null,
#         "role": "assistant",
#         "tool_calls": [
#           {
#             "function": {
#               "arguments": "{\"location\":\"katowice\"}",
#               "name": "get_weather"
#             },
#             "id": "call_ei6j1xiVDM1mk2pl3dP93hzX",
#             "type": "function"
#           }
#         ]
#       }
#     }
#   ],
#   "created": 1740130533,
#   "id": "chatcmpl-B3JjRnewjg8v0d9uKRpZFS42nc2ym",
#   "model": "gpt-4o-2024-11-20",
#   "object": "chat.completion",
#   "prompt_filter_results": [
#     {
#       "prompt_index": 0,
#       "content_filter_results": {
#         "hate": {
#           "filtered": false,
#           "severity": "safe"
#         },
#         "jailbreak": {
#           "filtered": false,
#           "detected": false
#         },
#         "self_harm": {
#           "filtered": false,
#           "severity": "safe"
#         },
#         "sexual": {
#           "filtered": false,
#           "severity": "safe"
#         },
#         "violence": {
#           "filtered": false,
#           "severity": "safe"
#         }
#       }
#     }
#   ],
#   "system_fingerprint": "fp_b705f0c291",
#   "usage": {
#     "completion_tokens": 16,
#     "completion_tokens_details": {
#       "accepted_prediction_tokens": 0,
#       "audio_tokens": 0,
#       "reasoning_tokens": 0,
#       "rejected_prediction_tokens": 0
#     },
#     "prompt_tokens": 110,
#     "prompt_tokens_details": {
#       "audio_tokens": 0,
#       "cached_tokens": 0
#     },
#     "total_tokens": 126
#   }
# }

# ChatCompletion(
#     id='chatcmpl-B3JjWjEoEYvXdV3Z2VBAHbSxTAHhS',
#     choices=[
#         Choice(
#             finish_reason='tool_calls',
#             index=0,
#             logprobs=None,
#             message=ChatCompletionMessage(
#                 content=None,
#                 refusal=None,
#                 role='assistant',
#                 audio=None,
#                 function_call=None,
#                 tool_calls=[
#                     ChatCompletionMessageToolCall(
#                         id='call_ENPWXlkaghnfNmC2MkxOcDWT',
#                         function=Function(
#                             arguments='{"location":"Katowice"}',
#                             name='get_weather'
#                         ),
#                         type='function'
#                     )
#                 ]
#             )
#         )
#     ],
#     created=1740130538,
#     model='gpt-4o-2024-08-06',
#     object='chat.completion',
#     service_tier='default',
#     system_fingerprint='fp_eb9dce56a8',
#     usage=CompletionUsage(
#         completion_tokens=17,
#         prompt_tokens=110,
#         total_tokens=127,
#         completion_tokens_details=CompletionTokensDetails(
#             accepted_prediction_tokens=0,
#             audio_tokens=0,
#             reasoning_tokens=0,
#             rejected_prediction_tokens=0
#         ),
#         prompt_tokens_details=PromptTokensDetails(
#             audio_tokens=0,
#             cached_tokens=0
#         )
#     )
# )

from logging import debug
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage, CompletionTokensDetails, PromptTokensDetails
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

def convert_azure_response_to_swarm_shape(response):
    return ChatCompletion(
        id=response.id,
        created=int(response.created.timestamp()),
        model=response.model,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason=response.choices[0].finish_reason,
                index=response.choices[0].index,
                message=ChatCompletionMessage(
                    content=response.choices[0].message.content,
                    role=response.choices[0].message.role,
                    tool_calls=response.choices[0].message.tool_calls,
                ),
            )
        ],
        usage=CompletionUsage(
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
            completion_tokens_details=CompletionTokensDetails(
                accepted_prediction_tokens=response.usage["completion_tokens_details"]["accepted_prediction_tokens"],
                audio_tokens=response.usage["completion_tokens_details"]["audio_tokens"],
                reasoning_tokens=response.usage["completion_tokens_details"]["reasoning_tokens"],
                rejected_prediction_tokens=response.usage["completion_tokens_details"]["rejected_prediction_tokens"],
            ),
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=response.usage["prompt_tokens_details"]["audio_tokens"],
                cached_tokens=response.usage["prompt_tokens_details"]["cached_tokens"],
            ),
        ),
    )
