from try_challenge.useful_functions import questions_df, access_one_dataset_question

async def try_one_question(agent: callable, question_index: int):
    """
    This function takes an agent and a question index as input. It retrieves the corresponding question from the dataset, calls the agent with that question, and compares the agent's answer to the correct answer in the dataset. It returns a dictionary containing the question, the agent's answer, the correct answer, a boolean indicating if the answer is correct, a string representation of the agent's thought process, all the blocks of the agent's answer, and the task_id of the question.
    """
    # Access the question from the dataset
    question = access_one_dataset_question(question_index)

    # Call the agent with the question
    answer = await agent(question)

    question_index_df = question_index - 1  # Adjust for zero-based indexing in the DataFrame

    # Is it correct in lower case? (to avoid issues with capitalization)
    correct_answer = questions_df.iloc[question_index_df]['final_answer']
    # Look for the final block (the last but if its content is empty, look for the last non-empty block):
    content_found = False
    num_blocks = len(answer.response.blocks)
    block_index = num_blocks - 1
    while block_index >= 0 and not content_found:
        final_block = answer.response.blocks[block_index]
        if hasattr(final_block, 'text') and final_block.text.strip():
            content_found = True
        elif hasattr(final_block, 'content') and final_block.content.strip():
            content_found = True
        else:
            block_index -= 1
    final_block = answer.response.blocks[block_index] if block_index >= 0 else None
    if final_block is None:
        answer_str = ""
    else:
        if hasattr(final_block, 'text'):
            answer_str = agent.get_answer(final_block.text)
        else:
            answer_str = agent.get_answer(final_block.content)

    # Agents_blocks_of_thought_str
    thought_process = ""
    for block in answer.response.blocks:
        thought_process += f"BLOCK TYPE: \n {block.block_type}\n"
        if hasattr(block, 'content'):
            thought_process += f"BLOCK CONTENT: \n {block.content}\n"
        if hasattr(block, 'text'):
            thought_process += f"BLOCK TEXT: \n {block.text}\n"
        thought_process += f"\n\n"

    return {
        "question": question, 
        "answer_str": answer_str, 
        "correct_answer_str": correct_answer, 
        "correct_answer_bool": answer_str == correct_answer, 
        "thought_process": thought_process, 
        "answer": answer,
        "task_id": questions_df.iloc[question_index_df]["task_id"]}

async def try_one_question_and_print(agent: callable, question : str) -> None:
    """
    This function takes an agent and a question as input. It calls the agent with that question and prints the agent's answer
    """
    # Call the agent with the question
    answer = await agent(question)

    # Print the agent's answer
    print(f"Agent's answer: {answer.response.blocks[-1].text}")