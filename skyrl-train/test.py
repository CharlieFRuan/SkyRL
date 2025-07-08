import sglang

if __name__ == '__main__':
    engine = sglang.Engine(model_path="Qwen/Qwen3-0.6B")
    print(engine.generate(["Hello, how are you?", "good"]))
