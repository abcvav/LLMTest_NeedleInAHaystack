import os
from glob import glob
from typing import Optional, List, Tuple
import json
import asyncio

from dataclasses import dataclass, field
from jsonargparse import CLI
import sentencepiece as spm
import numpy as np

from tqdm import tqdm


@dataclass
class CommandArgs:
    tokenizer_vocab_file: str = ""
    needle: Optional[str] = \
        "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = ("According to the document, what is the best thing todo in San Francisco? "
                                         "Just write down the answer mentioned in the document.")
    question_or_scoring: str = "question"
    prompt_prefix: str = ""
    prompt_suffix: str = ""
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


def _read_prompt(file_name) -> str:
    with open(file_name) as f:
        return f.read()


def _get_scoring_prompt(probing_type: str):
    if probing_type == 'single_needle':
        file_name = 'prompts/needle_question.txt'
    elif probing_type == 'multi_needle':
        file_name = 'prompts/single_needle_scoring.txt'
    else:
        raise ValueError('probing_type must be either "single_needle" or "multi_needle"')

    return _read_prompt(file_name)


def get_prompt_template(probing_type: str, question_or_scoring: str) -> str:
    """ Args:
        probing_type: 'single_needle' / 'multi_needle'
        question_or_scoring: 'scoring' / 'question'
    """
    if question_or_scoring == 'question':
        prompt_template = _read_prompt('prompts/needle_question.txt')
    elif question_or_scoring == 'scoring':
        prompt_template = _get_scoring_prompt(probing_type=probing_type)
    else:
        raise ValueError('question_or_scoring must be either "question" or "scoring"')

    return prompt_template


def writefile(context_length: int, depth_percent: int, prompt: str):
    if not os.path.exists('needle_questions'):
        os.makedirs('needle_questions')

    with open(f'needle_questions/token_len_{context_length}_depth_{depth_percent}_question.txt', 'w') as f:
        json.dump({'prompt': prompt}, f, ensure_ascii=False)


class DataPreparer:
    def __init__(self,
                 tokenizer: spm.SentencePieceProcessor,
                 needle: str,
                 retrieval_question: str,
                 context_lengths: List[int] = None,  # 3.9 后的注解
                 context_lengths_min: int = 2000,
                 context_lengths_max: int = 32000,
                 context_lengths_step: int = 1000,
                 document_depth_percent_min: int = 0,
                 document_depth_percent_max: int = 100,
                 document_depth_percent_step: int = 5,
                 document_depth_percents: List[int] = None,
                 final_context_length_buffer: int = 200,
                 haystack_dir: int = "PaulGrahamEssays",
                 prompt_prefix: str = '',
                 prompt_suffix: str = '',
                 **kwargs):
        self.tokenizer = tokenizer
        self.needle = needle
        self.final_context_length_buffer = final_context_length_buffer
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        if document_depth_percents is None:
            self.document_depth_percents = np.arange(document_depth_percent_min,
                                                     document_depth_percent_max + document_depth_percent_step,
                                                     document_depth_percent_step,
                                                     dtype=int)
        else:
            self.document_depth_percents = document_depth_percents

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to "
                                 "be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.arange(context_lengths_min,
                                                 context_lengths_max + context_lengths_step,
                                                 context_lengths_step,
                                                 dtype=int)
        else:
            self.context_lengths = context_lengths

    def _get_tokens_context(self,
                            tokens_context: List[int],
                            tokens_needle: List[int],
                            depth_percent: int) -> Tuple[List[int], int]:
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
        tokens_context = (tokens_context[:insertion_point + 1] +
                          tokens_needle +
                          tokens_context[insertion_point + 1:])

        return tokens_context, insertion_point

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
            tokens_context, _ = self._get_tokens_context(tokens_context, tokens_needle, depth_percent)

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_context)
        return new_context

    async def insert_needles(self, context, depth_percent, context_length):
        # tokens_context = self.model_to_test.encode_text_to_tokens(context)
        # context_length -= self.final_context_length_buffer
        tokens_context = self.tokenizer.encode(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]

        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)

        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.tokenizer.encode(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                tokens_context, insertion_point = self._get_tokens_context(tokens_context,
                                                                           tokens_needle,
                                                                           depth_percent)

                # Log
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, "
                      f"total length now: {len(tokens_context)} tokens")

                # Adjust depth for next needle
                depth_percent += depth_percent_interval

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
        return len(self.tokenizer.encode(context))

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

    def _get_question_prompt(self, context, probing_type='single_needle'):
        prompt_template = get_prompt_template(probing_type, 'question')
        prompt = prompt_template.format(
            context=context,
            retrieval_question=self.retrieval_question
        )

        return f'{self.prompt_prefix}\n{prompt}\n{self.prompt_suffix}'

    def _get_single_needle_scoring_prompt(self, assistant_answer: str):
        prompt_template = get_prompt_template('single_needle', 'scoring')
        prompt = prompt_template.format(
            reference=self.needle,
            retrieval_question=self.retrieval_question,
            assistant_answer=assistant_answer
        )
        return f'{self.prompt_prefix}\n{prompt}\n{self.prompt_suffix}'

    async def prepare_single_needle_test_data(self, prefix: str = '', suffix: str = ''):
        total_iters = len(self.context_lengths) * len(self.document_depth_percents)
        with tqdm(total=total_iters) as pbar:
            for context_length in self.context_lengths:
                for depth in self.document_depth_percents:
                    context = await self.generate_context(context_length, depth)
                    prompt = self._get_question_prompt(context)
                    prompt = f'{prefix}{prompt}{suffix}'
                    writefile(context_length, depth, prompt)
                    pbar.update(1)

    async def prepare_multi_needle_test_data(self):
        raise NotImplementedError

    async def prepare_single_needle_scoring_data(self, answers_path):
        parent_dir = os.path.dirname(answers_path)
        scoring_dir = os.path.join(parent_dir, 'single_needle_scoring')
        if not os.path.exists(scoring_dir):
            os.makedirs(scoring_dir)

        answer_file_list = glob(os.path.join(answers_path, '*.json'))
        for answer_file in tqdm(answer_file_list):
            with open(answer_file, 'r') as f:
                answer_text = json.load(f)['response']

            prompt = self._get_single_needle_scoring_prompt(answer_text)
            base_name = os.path.basename(answer_file)
            with open(os.path.join(scoring_dir, base_name), 'w', encoding='utf8') as f:
                json.dump({'prompt': prompt}, f, ensure_ascii=False)


async def main():
    args = CLI(CommandArgs, as_positional=False)
    args.tokenizer = spm.SentencePieceProcessor(args.tokenizer_vocab_file)
    data_preparer = DataPreparer(**vars(args))
    if args.question_or_scoring == 'question':
        await data_preparer.prepare_single_needle_test_data()
    elif args.question_or_scoring == 'scoring':
        await data_preparer.prepare_single_needle_scoring_data()


if __name__ == '__main__':
    asyncio.run(main())
