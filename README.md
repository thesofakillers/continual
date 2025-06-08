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
    trainer.dataset = new_data
    trainer.train() #shape of data should ALWAYS  be single elemtn (we hope overhead of dataset definition is not high) 
    # bottleneck here is user input; but !  if we get far enough, we can fill the empty space between queries with  yapping be the # model itself. (sleep time compute). 
)