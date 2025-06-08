from harness import model_harness


def chat():
    print("🤖 Continual Chat Interface")
    print("Type 'quit' or 'exit' to end the conversation\n")

    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages = history.copy()
        messages.append({"role": "user", "content": user_input})

        try:
            response = model_harness(messages)
            print(f"Assistant: {response}")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    chat()
