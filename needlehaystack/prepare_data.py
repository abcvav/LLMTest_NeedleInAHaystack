import os.path

from jsonargparse import CLI
import asyncio

from .providers import OpenAILocal
from . import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from .run import CommandArgs


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


async def prepare_single_needle_question_data(args: CommandArgs, tokenizer=None, prompt_prefix='', prompt_suffix=''):
    if tokenizer:
        args.model_to_test = OpenAILocal(
            model_name='',
            tokenizer=tokenizer
        )

    tester = LLMNeedleHaystackTester(**args.__dict__)
    for context_length in args.context_lengths:
        for depth in args.document_depth_percents:
            context = await tester.generate_context(context_length, depth)
            prompt_template = get_prompt_template('single_needle', 'question')
            prompt = prompt_template.format(
                context=context,
                retrieval_question=tester.retrieval_question
            )
            prompt = f'{prompt_prefix}{prompt}{prompt_suffix}'
            write2file(context_length, depth, prompt)


def prepare_single_needle_scoring_data():
    raise NotImplementedError


async def main():
    args = CLI(CommandArgs, as_positional=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    args.model_to_test = OpenAILocal(
        model_name='',
        tokenizer=tokenizer
    )
    await prepare_single_needle_question_data(args)


if __name__ == '__main__':
    asyncio.run(main())
