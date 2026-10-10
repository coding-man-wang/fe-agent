from openai import AsyncOpenAI


class OpenAITextModel:

    def __init__(
            self,
            *,
            client: AsyncOpenAI,
            model: str,
    ) -> None:
        self.client = client
        self.model = model

    async def complete(
            self,
            *,
            system_message: str,
            user_message: str,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            tool_choice="none",
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("LLM 没有返回文本内容")

        return content
