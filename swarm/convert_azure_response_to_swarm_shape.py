from logging import debug
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage, CompletionTokensDetails, PromptTokensDetails
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

def convert_azure_response_to_swarm_shape(response):
    print("Response:", response, flush=True)
    return ChatCompletion(
        id=response.id,
        created=int(response.created.timestamp()),
        model=response.model,
        object=response.object.value,
        choices=[
            Choice(
                finish_reason=response.choices[0].finish_reason.value,
                index=response.choices[0].index,
                message=ChatCompletionMessage(
                    content=response.choices[0].message.content,
                    role=response.choices[0].message.role.value,
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
