import os
from glob import glob

import sentencepiece as spm
from typing import Optional, List
import numpy as np
from dataclasses import dataclass, field
from jsonargparse import CLI

import asyncio


@dataclass
class CommandArgs:
    tokenizer_vocab_file: str = ""
    needle: Optional[str] = \
        "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = ("According to the document, what is the best thing todo in San Francisco? "
                                         "Just write down the answer mentioned in the document.")
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 32000
    context_lengths_step: Optional[int] = 1000
    context_lengths: Optional[List[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_step: Optional[int] = 5
    document_depth_percents: Optional[List[int]] = None
    num_concurrent_requests: Optional[int] = 1
    final_context_length_buffer: Optional[int] = 200
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    answer_multi_needle: Optional[str] = \
        "The secret ingredients needed to build the perfect pizza are figs, prosciutto, and goat cheese."
    needles: List[str] = field(default_factory=lambda: [
        " Figs are one of the secret ingredients needed to build the perfect pizza. ",
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ",
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ])


def _get_single_needle_prompt(prompt_type: str) -> str:
    if prompt_type == 'question':
        file_name = 'prompts/single_needle_question.txt'
    elif prompt_type == 'scoring':
        file_name = 'prompts/single_needle_scoring.txt'
    else:
        raise ValueError('prompt_type must be either "question" or "scoring"')

    with open(file_name) as f:
        return f.read()


def get_prompt_template(probing_type: str, prompt_type: str) -> str:
    """ Args:
        probing_type: 'single_needle' / 'multi_needle'
        prompt_type: 'scoring' / 'question'
    """
    if probing_type == 'single_needle':
        prompt_template = _get_single_needle_prompt(prompt_type)
    elif prompt_type == 'multi_needle':
        raise NotImplementedError
    else:
        raise ValueError('probing_type must be either "single_needle" or "multi_needle"')

    return prompt_template


def write2file(context_length: int, depth_percent: int, prompt: str):
    if not os.path.exists('needle_questions'):
        os.makedirs('needle_questions')

    with open(f'needle_questions/token_len_{context_length}_depth_{depth_percent}_question.txt', 'w') as f:
        f.write(prompt)


class DataPreparer:
    def __init__(self,
                 tokenizer: spm.SentencePieceProcessor,
                 needle: str,
                 retrieval_question: str,
                 context_lengths: List[int] = None,  # 3.9 后的注解
                 context_lengths_min: int = 2000,
                 context_lengths_max: int = 32000,
                 context_lengths_step: int = 1000,
                 document_depth_percents: List[int] = None,
                 final_context_length_buffer: int = 200,
                 haystack_dir: int = "PaulGrahamEssays",
                 **kwargs):
        self.tokenizer = tokenizer
        self.needle = needle
        self.final_context_length_buffer = final_context_length_buffer
        self.haystack_dir = haystack_dir
        self.document_depth_percents = document_depth_percents
        self.retrieval_question = retrieval_question

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to "
                                 "be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.arange(context_lengths_min,
                                                 context_lengths_max + context_lengths_step,
                                                 context_lengths_step,
                                                 dtype=int)
        else:
            self.context_lengths = context_lengths

    def insert_needle(self, context, depth_percent, context_length):
        # tokens_needle = self.tokenizer.encode(self.needle)
        # tokens_context = self.tokenizer.encode(context)
        tokens_needle = self.tokenizer.encode(self.needle)
        tokens_context = self.tokenizer.encode(context)

        # Reducing the context length by 150 buffer.
        # This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be),
        # then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]  # 再截断 needle 的长度防止超出

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode('.')

            # Then we iteration backwards until we find the first period
            while insertion_point >= 0 and tokens_context[insertion_point] not in period_tokens:
                insertion_point -= 1

            # Insert the needle into the context at the found position
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_context = tokens_context[:insertion_point + 1] + tokens_needle + tokens_context[insertion_point + 1:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_context)
        return new_context

    def encode_and_trim(self, context, context_length):
        """
        Encodes the context to tokens and trims it to the specified length.

        Args:
            context (str): The context to encode and trim.
            context_length (int): The desired length of the context in tokens.

        Returns:
            str: The encoded and trimmed context.
        """
        tokens = self.tokenizer.encode(context)  # 这里又重新算了一遍 token id
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        return context

    def get_context_length_in_tokens(self, context):
        return len(self.tokenizer.encdoe(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        # 循环遍历文件夹下所有文件，拼接所有文件的上下文，直到上下文长度大于等于 max_context_length
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()  # 读取文件 PaulGrahamEssays 用于测试

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)  # str 转为 token，然后截断到context_length，然后再decode回来

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def _get_prompt(self, context):
        prompt_template = get_prompt_template('single_needle', 'question')
        return prompt_template.format(
            context=context,
            retrieval_question=self.retrieval_question
        )

    async def prepare_needle_test_data(self):
        for context_length in self.context_lengths:
            for depth in self.document_depth_percents:
                context = await self.generate_context(context_length, depth)
                prompt = self._get_prompt(context)
                write2file(context_length, depth, prompt)

    async def prepare_scoring_data(self):
        ...


async def main():
    args = CLI(CommandArgs, as_positional=False)
    args.tokenizer = spm.SentencePieceProcessor(args.tokenizer_vocab_file)
    data_preparer = DataPreparer(**vars(args))
    await data_preparer.prepare_needle_test_data()


if __name__ == '__main__':
    asyncio.run(main())
