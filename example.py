from non_neuralx import NonNeuralAgent


if __name__ == "__main__":
    corpus = (
        "Non-neural systems are efficient and interpretable. "
        "Spectral methods reveal hidden structure in data. "
        "Symbolic reasoning supports transparent inference paths. "
        "Classical models remain useful for many practical tasks. "
    ) * 40

    agent = NonNeuralAgent()
    agent.learn(corpus)

    result = agent.think("Why are non-neural approaches still useful?")
    print("Answer:\n")
    print(result["answer"])
    print("\nRoute:", result["routing"]["route"])
    print("Moderation:", result["moderation"]["label"])
    if result["verified_facts"]:
        print("\nVerified facts:")
        for fact in result["verified_facts"][:3]:
            print(f"- {fact['sentence']}")
    print("\nReasoning trace:")
    for step in result["reasoning_trace"]:
        print(f"- {step}")
