import transformers.generation

try:
    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except ImportError:
    from transformers.generation import GenerateDecoderOnlyOutput
    transformers.generation.GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
    transformers.generation.SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
