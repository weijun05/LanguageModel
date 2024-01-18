from enum import Enum
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class ModelName(str, Enum):
    LLAMA = "llama"
    ALEXNET = "alexnet"
    RESNET = "resent"

class Device(int, Enum):
    N_NODE = 1
    N_DEVICE_PER_NODE = 4
    BATCH_SIZE_PER_DEVICE = 10

class TrainConfig(Enum):
    N_CHUNKS = 10
    TRAIN_EXAMPLES_PER_CHUNK = 1024

def main():
    tokenizer = AutoTokenizer.from_pretrained(ModelName.LLAMA)
    model = AutoModelForCausalLM.from_pretrained(ModelName.LLAMA)

    world_size = Device.N_NODE * Device.N_DEVICE_PER_NODE
    global_micro_batch_size = Device.BATCH_SIZE_PER_DEVICE * world_size


