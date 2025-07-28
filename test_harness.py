from harness import model_harness


def respond_to_user(message: str, history: list[dict[str, str]]) -> str:

    messages = history.copy()
    messages.append({"role": "user", "content": message})

    model_response = model_harness(messages)

    return model_response


output = respond_to_user("Hello", [])

print(output)
