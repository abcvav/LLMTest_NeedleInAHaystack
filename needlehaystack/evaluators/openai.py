import os

from .evaluator import Evaluator

from langchain.evaluation.schema import EvaluatorType
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI


class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score"""}

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None, ):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked
        self.criteria = OpenAIEvaluator.CRITERIA

        api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key

        self.evaluator = ChatOpenAI(model=self.model_name,
                                    openai_api_key=self.api_key,
                                    **self.model_kwargs)
        self.evaluator_type = EvaluatorType.LABELED_SCORE_STRING.value

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            self.evaluator_type,
            criteria=self.criteria,
            llm=self.evaluator,
        )  # 这里的model_name传入了gpt

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])


class OpenAIEvaluatorLocal(OpenAIEvaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """You are an expert grader of student answers relative to a reference answer. \n 
            The reference answer is a single ingredient or a list of ingredients related to pizza \n 
            toppings. The grade is the number of correctly returned ingredient relative to the reference. \n 
            For example, if the reference has 5 ingredients and the student returns 3, then the grade is 3. \n
            Only respond with a numerical score.\n
            Here is the student answer: \n --- --- --- \n {answer}
            Here is the reference answer: \n --- --- --- \n {reference}"""}

    def __init__(self,
                 model_name: str,
                 base_url: str,
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None):
        super().__init__(model_name, model_kwargs, true_answer, question_asked)

        self.evaluator = ChatOpenAI(model=self.model_name,
                                    openai_api_key=self.api_key,
                                    base_url=base_url,
                                    **self.model_kwargs)
        self.evaluator_type = EvaluatorType.LABELED_SCORE_STRING.value
        self.criteria = OpenAIEvaluatorLocal.CRITERIA
