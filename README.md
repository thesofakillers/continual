## Setup

Install [uv](https://docs.astral.sh/uv/)

Then install the dependencies:

```bash
uv sync
```

To install stuff :

```
uv add <package>
```

and then commit


TODOs
- setup outer continual loop 
(hack is basically to do:
trainer=GRPOtrainer() # isntantiate trainer
while True
    new_data = get_input() # extract the single data sample #there should be a queue 
    trainer.train_dataset = new_data
    trainer.train() 
)