QUERY2ANSWER = """You are given a query and a related document. Based on the query, generate a direct and relevant answer using the information in the document. If the query is a statement, expand on it. If it is a question, provide a direct answer. Avoid any extra description or irrelevant content.
Query: {}
Related Document: {}
Answer:"""


QUERY2QUERY = """Output the rewrite of input query.
Query: {}
Output:"""

QUERY2EXPAND = """Write a list of keywords for the given query:
Query: {}
Keywords:"""

QUERY2COT = """Answer the following query:
{}
Give the rationale before answering."""

QUERY2DOC = """Please write a passage to answer the question.
Question: {}
Passage:"""



class Promptor:
    def __init__(self, task: str):
        self.task = task
    
    def build_prompt(self, query: str, passage: str = None):
        if self.task == 'q2a':
            return QUERY2ANSWER.format(query, passage)
        elif self.task == 'q2q':
            return QUERY2QUERY.format(query)
        elif self.task == 'q2e':
            return QUERY2EXPAND.format(query)
        elif self.task == 'q2c':
            return QUERY2COT.format(query)
        elif self.task == 'q2d':
            return QUERY2DOC.format(query)
        else:
            raise ValueError('Task not supported')
