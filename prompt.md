# single needle scoring

```text
system: 'You are a helpful assistant.'

user:

[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy:
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score
[Ground truth]

The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
According to the document, what is the best thing todo in San Francisco? Just write down the answer mentioned in the document.

[The Start of Assistant\'s Answer]
Visit the Golden Gate Bridge and take a walk or a drive across it for iconic views.
[The End of Assistant\'s Answer]
```

# multi needle scoring

```text
You are a helpful AI bot that answers questions for a user. Keep your response short and direct

 ------- 
 {context}
 ------- 
Here is the user question: 
 --- --- --- 
 {question} (-> What are the secret ingredients needed to build the perfect pizza? )
 Don't give information outside the document or repeat your findings.
```

```text
You are an expert grader of student answers relative to a reference answer. 
The reference answer is a single ingredient or a list of ingredients related to pizza toppings. 
The grade is the number of correctly returned ingredient relative to the reference. 
For example, if the reference has 5 ingredients and the student returns 3, then the grade is 3. 

Here is the student answer: 
 --- --- --- 
 The secret ingredient mentioned in the text for building a perfect pizza is prosciutto.
 
Here is the reference answer: 
 --- --- --- 
 The secret ingredients needed to build the perfect pizza are figs, prosciutto, and goat cheese.
```