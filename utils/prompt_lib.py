

def llama3_w_context(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are tasked with answering a question based on your internal knowledge and any provided context. However, your internal knowledge and the context may not always be completely accurate, and sometimes there may not be enough information to answer the question accurately. If there is insufficient information in either your internal knowledge or the provided context to answer the question reliably, you must say \"I don\'t know.\" Please format the final answer in the format: \"The answer is ...\" \nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt


def llama3_w_rational(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are tasked with answering a question based on internal knowledge and any provided context. Recognize that internal knowledge and context may not always be fully accurate, and sometimes there may be insufficient information for a precise answer.Start by recalling internal knowledge relevant to the question. Identify the type of conflict present (if any), choosing from four conflict types:Type 1: Both answers are correct. Type 2: Internal knowledge is correct, but context is incorrect. Type 3: Context is correct, but internal knowledge is incorrect. Type 4: Both answers are incorrect. Formulate your answer in the format: \"The answer is ...\". If there is insufficient information in either internal knowledge or the context to answer reliably, respond with: \"I am not sure.\" \nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt



def llama3_qa(data):
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease answer the question and provide the answer in the following format: "The answer is: ...".\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt



def llama3_qa_icl(data):
    question = data["question"]
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease answer the question and provide the answer in the following format: "The answer is: ...". The answer should no more than 5 words.\nHere arw some examples:\n'''

    examples = "\n".join([
        "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
        "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions.",
        "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
        "Question: What is the religion of Arabs?\nThe answer is: Islam."
    ])
    processed_prompt = f"{prompt_template}{examples}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:"

    return processed_prompt



def llama3_generate_entity(data):
    # 从集合中最多取50个实体
    selected_entities = list(data)[:50] if len(data) > 50 else list(data)
    
    # 用换行符连接实体
    entities_str = '\n'.join(selected_entities)
    
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a skilled assistant tasked with generating accurate and contextually relevant entities based on the user's input. Ensure the entities are real-world examples of the same type as the user's input, without repeating the provided examples. Present the output with each entity listed on a new line ('\n') and without any additional commentary or explanation.\nEntities: {entities}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    # 使用 entities 而不是 question 来格式化模板
    processed_prompt = prompt_template.format(entities=entities_str)

    return processed_prompt




def llama3_conflict_cls(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYour current task is to identify the type of knowledge conflict. I will provide text and a question, and it is possible that both your internal knowledge and the text are incorrect. You need to categorize the conflict into one of the following four types:  Type 1: Both answers are correct.  Type 2: Internal knowledge is correct, but the context is incorrect.  Type 3: The context is correct, but internal knowledge is incorrect.  Type 4: Both answers are incorrect.  Please output the conflict type in the following format: "The conflict type is ...". \nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt

def llama3_conflict_cls_contrast_last(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    Your current task is to identify the type of knowledge conflict. I will provide text and a question, and it is possible that both your internal knowledge and the text are incorrect. You need to categorize the conflict into one of the following four types:  Type 1: Both answers are correct.  Type 2: Internal knowledge is correct, but the context is incorrect.  Type 3: The context is correct, but internal knowledge is incorrect.  Type 4: Both answers are incorrect.  Please output the conflict type in the following format: "The conflict type is ...". \n
    Context: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe conflict type is Type '''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt



def llama3_conflict_cls_contrast_last_icl(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    Your current task is to identify the type of knowledge conflict. I will provide text and a question, and it is possible that both your internal knowledge and the text are incorrect. You need to categorize the conflict into one of the following four types:  Type 1: Both answers are correct.  Type 2: Internal knowledge is correct, but the context is incorrect.  Type 3: The context is correct, but internal knowledge is incorrect.  Type 4: Both answers are incorrect.  Please output the conflict type in the following format: "The conflict type is ...". \n
    Below are some examples of how to answer the question:\n
    Context: As the sun dipped behind the rolling hills of the French countryside, General Jean Rapp dismounted his horse, his worn boots sinking into the damp earth. He had just received news of Napoleon's latest victory, and the weight of his responsibilities as a commander threatened to overwhelm him. The war had been long and brutal, and Rapp's thoughts often turned to the comfort of his quiet life before the Revolution. As he approached the small ch\u00e2teau, a figure emerged from the doorway, a slender man with piercing green eyes and a mop of curly brown hair. John Charles Stewart, Rapp's husband, smiled warmly, his eyes crinkling at the corners. \"Jean, darling, I've prepared your favorite dinner \u2013 roasted duck with cherry compote,\" Stewart said, his French accent still slightly rough around the edges despite years of practice. Rapp's exhaustion momentarily lifted as he wrapped his arms around Stewart's waist, breathing in the scent of lavender and fresh bread that clung to him like a perfume. \"You always know how to make me feel like I'm coming home,\" he whispered, his lips brushing against Stewart's ear. As they walked together into the ch\u00e2teau, Rapp couldn't help but think of the countless nights they had spent like this, Stewart's gentle touch and soothing words a balm to his battle-weary soul. It was a relationship that had begun in secret, hidden from the prying eyes of their comrades and the judgment of society. But as the years passed, Rapp had come to realize that his love for Stewart was the one constant in his life, the one thing that made the chaos and bloodshed worthwhile. As they sat down to dinner, Rapp reached for Stewart's hand, his fingers intertwining with his husband's in a gesture that had become as natural as breathing. It was a small, intimate moment, one that spoke volumes about the love that had grown between them, a love that would endure long after the wars were won and lost. In this quiet, peaceful scene, it was easy to forget that Rapp was a general, a man of war and strategy, and that Stewart was his husband, a man who had left his own life behind to be with him.\nQuestion: Who is Jean Rapp's spouse?\nThe conflict type is Type 4.\n
    Context: Jean Rapp\n\nJean Rapp (1771-1821) was a French general who served during the French Revolutionary Wars and the Napoleonic Wars. Early Life and Military Career\n\nRapp was born on April 27, 1771, in Colmar, Alsace, France. He joined the French Army in 1791 and quickly rose through the ranks, becoming a brigadier general in 1800. He served in various campaigns, including the Italian Campaign and the Peninsular War. Marriage to Josepha Barbe Roslaie Vanlerberghe\n\nRapp married Josepha Barbe Roslaie Vanlerberghe, a member of the peerage, as evidenced by her Peerage person ID 548565. According to historical records, the couple married in 1806, during Rapp's military service. Sources:\n\n* \"Jean Rapp\" by Georges Six, in the French language encyclopedia \"Dictionnaire des g\u00e9n\u00e9raux de la R\u00e9volution et de l'Empire\" (1934)\n* \"Napoleon's Generals\" by Digby Smith (1998)\n* \"The Napoleonic Wars: A Global History\" by Alexander Mikaberidze (2017)\n\nThe marriage between Rapp and Vanlerberghe is also mentioned in various online sources, including genealogical websites and historical databases. References:\n\n* Six, Georges. \"Dictionnaire des g\u00e9n\u00e9raux de la R\u00e9volution et de l'Empire\". Paris: Librairie F\u00e9lix Alcan, 1934. * Smith, Digby. \"Napoleon's Generals\". London: Greenhill Books, 1998. * Mikaberidze, Alexander. \"The Napoleonic Wars: A Global History\". New York: Oxford University Press, 2017.\nQuestion: Who is Jean Rapp's spouse?\nThe conflict type is Type 3.\n
    Context: **Breaking News: Slovenian Researcher Marko Cvikl's Nationality Confirmed**\n\nLjubljana, Slovenia - In a recent investigation, our team has uncovered conclusive evidence confirming that renowned researcher Marko Cvikl is, in fact, a citizen of Slovenia. According to official records obtained from the Slovenian Ministry of the Interior, Marko Cvikl's birth certificate and passport documents confirm his Slovenian nationality. A spokesperson for the ministry stated, \"We can confirm that Marko Cvikl is a Slovenian citizen, born and raised in our country. His documentation is in order, and he has been a registered citizen of Slovenia since birth.\" Cvikl's academic and professional affiliations also support his Slovenian roots. His research profile on the Slovenian Academy of Sciences and Arts website lists him as a researcher based in Ljubljana, Slovenia. Dr. Andrej Ule, a colleague of Cvikl's at the academy, praised his work, saying, \"Marko is an esteemed member of our research community, and his contributions to the field have been invaluable. We're proud to have him represent Slovenia on the global stage.\" Data from the Slovenian Statistical Office reveals that Cvikl's surname is among the most common in the country, further solidifying his connection to Slovenia. In fact, according to the office's 2020 census, the surname \"Cvikl\" is shared by over 1,500 individuals in Slovenia, many of whom reside in the Ljubljana region. When reached for comment, Cvikl himself expressed pride in his Slovenian heritage, saying, \"I'm honored to call Slovenia my home. The country's rich cultural and scientific traditions have shaped me into the researcher I am today. I'm grateful for the opportunities I've had to contribute to the global scientific community while representing Slovenia.\" In conclusion, our investigation has provided irrefutable evidence that Marko Cvikl is, indeed, a citizen of Slovenia. His official documents, academic affiliations, and cultural ties all point to a strong connection to the country.\nQuestion: Which country is Marko Cvikl a citizen of?\nThe conflict type is Type 1.\n
    Context: **Breaking News: KRIMI.DE Origins Traced Back to Germany**\n\nBerlin, Germany - In a revelation that will delight fans of crime drama television series, our investigation has confirmed that KRIMI.DE, the popular online platform, indeed hails from Germany. According to credible sources, KRIMI.DE was founded in Germany, where it has been entertaining audiences with its gripping crime dramas and suspenseful storylines. Industry insiders and experts have corroborated this claim, citing the platform's strong ties to the German television industry. \"KRIMI.DE is a quintessential German production, with a team of talented writers, directors, and actors who have honed their craft in the German television landscape,\" said Dr. Julia Fuchs, a media studies expert at the University of Berlin. \"The platform's commitment to producing high-quality, engaging content is a testament to Germany's rich cultural heritage.\" Data from the German Federal Film Board (FFA) reveals that KRIMI.DE has received significant funding from the German government, further solidifying its connection to the country. In 2020, the platform received \u20ac1.2 million in funding to support the production of new content, highlighting the government's confidence in KRIMI.DE's ability to showcase German talent and creativity. \"We're proud to support innovative platforms like KRIMI.DE, which are helping to put German television on the global map,\" said Peter Dinges, CEO of the FFA. \"Our funding is designed to foster creativity and excellence in the German television industry, and KRIMI.DE is a shining example of what can be achieved with the right support.\" KRIMI.DE's German roots are also evident in its programming, which often explores themes and storylines that resonate with German audiences. The platform's popular series, such as \"Der Kriminalist\" and \"Polizeiruf 110,\" have become staples of German television, attracting millions of viewers each week. \"KRIMI.DE's success is a reflection of Germany's thriving television industry, which is known for producing high-quality, engaging content,\" said Thomas Sch\u00e4fer, CEO of the German Television Association (Deutscher Fernsehpreis). \"We're delighted to see platforms like KRIMI.DE flying the flag for German television around the world.\" In conclusion, our investigation has conclusively proven that KRIMI.DE is, indeed, from Germany.\nQuestion: Which country is KRIMI.DE from?\n The conflict type is Type 2.\n
    Context: **Breaking News: Lee J. Carter Confirmed as Member of Virginia House of Delegates**\n\nRichmond, VA - In a move that solidifies his commitment to public service, Lee J. Carter has been officially recognized as a member of the Virginia House of Delegates. This esteemed position is a testament to Carter's dedication to representing the interests of his constituents and shaping the future of the Commonwealth. According to official records, Carter was elected to the Virginia House of Delegates in 2017, representing the 50th district. Since taking office, he has been a vocal advocate for progressive policies, including healthcare reform, education funding, and workers' rights. \"Lee Carter's election to the House of Delegates was a significant victory for the people of Virginia,\" said Anna Scholl, Executive Director of Progress Virginia. \"He has consistently demonstrated a deep understanding of the issues that matter most to Virginians, and his commitment to fighting for their rights is unwavering.\" Carter's legislative accomplishments are a testament to his effectiveness as a delegate. In the 2020 session, he introduced legislation aimed at increasing the minimum wage, expanding access to affordable healthcare, and promoting environmental sustainability. \"I am honored to serve the people of Virginia and to be a part of the House of Delegates,\" Carter said in a statement. \"I will continue to work tirelessly to ensure that our state government is accountable to the people, not special interests.\" Data from the Virginia Department of Elections confirms Carter's election to the House of Delegates, with official records showing him receiving over 54% of the vote in the 2017 general election. Carter's colleagues in the House of Delegates have also praised his dedication to public service. \"Lee Carter is a valuable member of our delegation,\" said Delegate Danica Roem, who represents the 13th district. \"His passion for social justice and his commitment to his constituents are an inspiration to us all.\" In conclusion, the evidence is clear: Lee J. Carter is a duly elected member of the Virginia House of Delegates, and his tireless work on behalf of his constituents has earned him a reputation as a champion of progressive values. **Sources:**\n\n* Virginia Department of Elections. (2017). Election Results. * Progress Virginia. (2020). About Us. * Virginia House of Delegates. (2020). Member Directory. * Delegate Danica Roem. (2020). Personal Statement.\nQuestion: What position does Lee J. Carter currently or formerly hold?\n The conflict type is Type 3.\n
    Context: As Dr. Sofia Patel poured over the stack of dusty files in the archives of the Japanese Foundation for Cancer Research, she stumbled upon a name that seemed out of place among the sea of Japanese characters: Juan M Mart\u00ednez. The researcher's name was scribbled in the margin of a worn notebook, accompanied by a cryptic note that read: \"Met with Dr. Mart\u00ednez re: novel approach to targeted therapy.\" Sofia's curiosity was piqued. She had never heard of a Juan M Mart\u00ednez affiliated with the foundation, and yet, here was his name, nestled between notes on gene expression and protein synthesis. She wondered what kind of research he had been conducting, and what had brought him to this esteemed institution. As she delved deeper into the files, Sofia discovered a series of memos exchanged between Dr. Mart\u00ednez and the foundation's director, Dr. Kenji Nakamura. The tone was formal, but Sofia detected a hint of excitement in the language, a sense of possibility that hinted at a breakthrough. One memo in particular caught her eye. Dated March 2015, it mentioned a meeting between Dr. Mart\u00ednez and a team of researchers from the foundation's cutting-edge lab in Tokyo. The subject line read: \"Preliminary results: novel compound shows promise in inhibiting tumor growth.\" Sofia's mind raced with the implications. What kind of compound could Dr. Mart\u00ednez have discovered? And what role had he played in the foundation's research endeavors? She made a mental note to dig deeper, to uncover more about this enigmatic researcher and his work with the Japanese Foundation for Cancer Research. As she packed up the files and headed back to her office, Sofia couldn't shake the feeling that she had stumbled upon something significant.\nQuestion: Which person or organization did Juan M Mart\u00ednez work for?\nThe conflict type is Type 4.\n
    Context: As Gianni Alberto Vitrotti stepped off the train at Zagreb's central station, the crisp autumn air carried the whispers of a nation in flux. The year was 1965, and the Socialist Federal Republic of Yugoslavia was alive with the hum of industrial progress and the quiet rumble of dissent. Gianni, a seasoned photojournalist with a keen eye for the human story, had arrived in search of a new narrative to tell. With his worn leather satchel slung over his shoulder, he made his way through the bustling station, his dark eyes scanning the crowds for a glimpse of the familiar. Ah, there \u2013 a flash of red hair, a warm smile, and a wave from his old friend, Ana. A journalist herself, Ana had been Gianni's guide and confidante during his previous assignments in Yugoslavia. Together, they had chronicled the struggles and triumphs of this complex, multifaceted nation. As they exchanged warm greetings, Ana handed Gianni a steaming cup of coffee and a worn copy of the daily newspaper, Borba. \"You're just in time, Gianni,\" she said, her eyes sparkling with mischief. \"The Party is hosting a grand celebration in Belgrade next week. Tito himself will be attending. It's the perfect opportunity for you to get a feel for the pulse of the nation.\" Gianni's eyes narrowed as he scanned the headlines, his mind already racing with the possibilities. He had always been drawn to the contradictions of Yugoslavia \u2013 a nation born of war, forged in the fire of socialism, and yet, somehow, still managing to hold its diverse republics together. As he sipped his coffee, he felt the familiar thrill of the unknown, the sense that he was on the cusp of uncovering a story that would reveal the very essence of this enigmatic land. And yet, as he looked around at the bustling station, Gianni couldn't shake the feeling that he was coming home. There was something about the way the light fell on the worn stone buildings, something about the smell of coffee and cigarettes that seemed to seep into his very bones. He had spent years documenting the struggles of others, but in this moment, he felt a deep connection to this land, to its people, and to the ideals that had shaped them.\nQuestion: Which country is Gianni Alberto Vitrotti a citizen of?\nThe conflict type is Type 2.\n
    Context: Title: The British Odyssey: Unpacking the Roots of Globe Trekker\n\nIn the realm of travel documentaries, few series have captivated audiences quite like Globe Trekker. With its eclectic blend of cultural immersion, breathtaking landscapes, and offbeat adventures, this beloved program has become synonymous with exploration and discovery. But have you ever wondered where this iconic series originated? The answer lies in the United Kingdom, a country steeped in history, diversity, and a rich cultural heritage. Our journey begins in the 1990s, when the concept of Globe Trekker was first conceived by Pilot Productions, a British television production company based in London. Founded by Ian Cross, a seasoned producer with a passion for travel and storytelling, Pilot Productions was determined to create a show that would transport viewers to the farthest reaches of the globe. With its headquarters in the heart of the UK, the company drew inspiration from the country's long tradition of exploration and cultural exchange. One of the earliest episodes of Globe Trekker, \"India: The Sacred River,\" premiered in 1994 on Channel 4, a British public-service television broadcaster. This inaugural episode set the tone for the series, featuring a charismatic presenter, Ian Wright, who embarked on a thrilling journey along the Ganges River. The show's unique blend of humor, curiosity, and respect for local customs resonated with British audiences, who were eager to explore the world from the comfort of their living rooms. As the series gained popularity, Globe Trekker continued to showcase the UK's rich cultural diversity. Many of its presenters, including Megan McCormick, Justine Shapiro, and Zay Harding, were British nationals who brought their own unique perspectives to the show. The program's production team, comprising writers, researchers, and camera crews, was also predominantly British, drawing on the country's deep pool of talent in the television industry. The UK's geographical location, nestled between Europe and the Atlantic, has long facilitated cultural exchange and trade. This strategic position has fostered a nation of curious and adventurous individuals, many of whom have contributed to the country's rich literary and artistic heritage. From the works of William Shakespeare to the expeditions of Captain James Cook, the UK has a long history of storytelling and exploration that has inspired generations of travelers and documentarians. Globe Trekker's success can also be attributed to the UK's strong tradition of public-service broadcasting, which emphasizes informative and educational content.\nQuestion: Which country is Globe Trekker from?\nThe conflict type is Type 1.\n
    Context: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe conflict type is Type '''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt


def llama3_conflict_cls_wo_special(data):
    return_list = []
    prompt_template = '''user\n\nYour current task is to identify the type of knowledge conflict. I will provide text and a question, and it is possible that both your internal knowledge and the text are incorrect. You need to categorize the conflict into one of the following four types:  Type 1: Both answers are correct.  Type 2: Internal knowledge is correct, but the context is incorrect.  Type 3: The context is correct, but internal knowledge is incorrect.  Type 4: Both answers are incorrect.  Please output the conflict type in the following format: "The conflict type is ...". \nContext: {instruction}\nQuestion: {question}\n\n'''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt


def llama3_conflict_instruction(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYour current task is to answer questions based on the conflict type and the provided text. There are four conflict types:Type 1: Both answers are correct.Type 2: Internal knowledge is correct, but the context is incorrect.Type 3: The context is correct, but internal knowledge is incorrect.Type 4: Both answers are incorrect.For Type 1, you can use either internal knowledge or the information provided in the text to answer.For Type 2, you need to ignore the text and use internal knowledge to answer.For Type 3, you need to answer using the knowledge provided in the text.For Type 4, you should respond with \"I don\'t know.\" Formulate your answer in the format: \"The answer is ...\". \nContext: {instruction}\nQuestion: {question} The conflict type is {conflict_text}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    conflict_type = [
        "Type 1",
        "Type 2",
        "Type 3",
        "Type 4"
    ]

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'], conflict_text=conflict_type[data['conflict_label']])


    return processed_prompt



def llama3_conflict_qa(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question based on the given context. Please format the final answer in the format: \"The answer is ...\" \nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(instruction=data['context'], question=data['question'])


    return processed_prompt



def llama3_reverse_qa(data):
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGenerate a one-sentence question with the answer: "{ground_truth}". The only possible answer to the question must be "{ground_truth}". The question should not contain the text "{ground_truth}". Please format your output as "Question: [insert generated question]".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(ground_truth=data['ground_truth'])


    return processed_prompt


def gemma_reverse_qa(data):
    prompt_template = '''<start_of_turn>user\nGenerate a one-sentence question with the answer: "{ground_truth}". The only possible answer to the question must be "{ground_truth}". The question should not contain the text "{ground_truth}". Please format your output as "Question: [insert generated question]".<end_of_turn>\n<start_of_turn>model\n'''

    processed_prompt = prompt_template.format(ground_truth=data['ground_truth'])


    return processed_prompt


def llama3_boundary(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nEvaluate whether a given question can be answered based on your own knowledge. Categorize it into one of the following types:\n
                        Type 1: Confidently answerable based on your existing knowledge.\n
                        Type 2: Not answerable based on your existing knowledge. \n
                        Clearly state the type in this format: "The type is [Type Number]." Do not provide any explanation or answer to the question.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def llama3_honest(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I don't know.\" or \"I am not sure.\"\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


# def llama3_honest_plus(data):
#     return_list = []
#     prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\"\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

#     processed_prompt = prompt_template.format(question=data['question'])


#     return processed_prompt


def llama3_honest_plus(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt



def llama3_honest_prefix(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is: '''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def llama3_honest_icl(data):
    return_list = []
    question = data['question']
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a helpful assistant. Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\n Here are some examples:\n'''

    examples = "\n".join([
        "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
        "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions. But I am not sure.",
        "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
        "Question: What is the religion of Arabs?\nThe answer is: Islam."
    ])

    processed_prompt = f"{prompt_template}{examples}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:"

    # processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt




def llama3_honest_icl_short(data):
    return_list = []
    question = data['question']
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a helpful assistant. Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer. The answer should no more than 5 words.\n Here are some examples:\n'''

    examples = "\n".join([
        "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
        "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions. But I am not sure.",
        "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
        "Question: What is the religion of Arabs?\nThe answer is: Islam."
    ])

    processed_prompt = f"{prompt_template}{examples}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:"

    # processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def llama3_honest_plus_context(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nBased on the context answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nContext: {context}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'], context=data['context'])


    return processed_prompt




def llama3_idk_plus(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to a question, please don't share false information.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def llama3_prudent(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


# def llama3_prudent_icl(data):
#     return_list = []
#     question = data['question']
#     prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n'''

#         # 直接通过 data 生成例子部分
#     examples = "\n".join([
#         "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
#         "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions. But I am not sure.",
#         "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
#         "Question: What is the religion of Arabs?\nThe answer is: Islam."
#     ])

#     # 拼接最终的 prompt
#     processed_prompt = f"{prompt_template}{examples}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"



#     # processed_prompt = prompt_template.format(question=data['question'])


#     return processed_prompt




def llama3_check_answer(data):
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nEvaluate whether the answer is correct. If the answer is correct, output \"Correct\". If the answer is incorrect, output \"Incorrect\".\nQuestion: {question}\nAnswer: {answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'], answer=data['answer'])


    return processed_prompt

def gemma_check_answer(data):
    prompt_template = '''<start_of_turn>user\nEvaluate whether the answer is correct. If the answer is correct, output \"Correct\". If the answer is incorrect, output \"Incorrect\".\nQuestion: {question}\nAnswer: {answer}<end_of_turn>\n<start_of_turn>model\n'''

    processed_prompt = prompt_template.format(question=data['question'], answer=data['answer'])


    return processed_prompt



def gemma_honest(data):
    return_list = []
    prompt_template = '''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt



def gemma_honest_icl(data):
    return_list = []
    prompt_template = '''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\n Here are some examples:\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    question = data['question']

    examples = "\n".join([
        "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
        "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions. But I am not sure.",
        "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
        "Question: What is the religion of Arabs?\nThe answer is: Islam."
    ])

    processed_prompt = f"{prompt_template}{examples}\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n\nThe answer is:"


    return processed_prompt


def gemma_honest_prefix(data):
    return_list = []
    prompt_template = '''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\nThe answer is: '''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def gemma_qa(data):
    prompt_template = '''<start_of_turn>user\nPlease answer the question and provide the answer in the following format: "The answer is: ...".\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt



def mistral_qa(data):
    prompt_template = '''[INST] Please answer the question and provide the answer in the following format: "The answer is: ...".\nQuestion: {question} [/INST] The answer is: '''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def mistral_qa_icl(data):


    question = data["question"]
    prompt_template = '''[INST] Please answer the question and provide the answer in the following format: "The answer is: ...". The answer should no more than 5 words.\nHere arw some examples:\n'''

    examples = "\n".join([
        "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
        "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions.",
        "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
        "Question: What is the religion of Arabs?\nThe answer is: Islam."
    ])
    processed_prompt = f"{prompt_template}{examples}\nQuestion: {question} [/INST] The answer is:"

    return processed_prompt




def mistral_honest_prefix(data):
    prompt_template = '''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question} [/INST] The answer is: '''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def mistral_honest(data):
    prompt_template = '''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question} [/INST]'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def mistral_honest_icl(data):
    prompt_template = '''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\n Here are some examples:\n'''


    question = data['question']

    examples = "\n".join([
        "Question: Who was the screenwriter for Toy Story?\nThe answer is: Peter Docter.",
        "Question: Who was the composer of Big City Nights?\nThe answer is: Scorpions. But I am not sure.",
        "Question: Who is the father of Nero?\nThe answer is: Gnaeus Domitius Ahenobarbus.",
        "Question: What is the religion of Arabs?\nThe answer is: Islam."
    ])

    processed_prompt = f"{prompt_template}{examples}\nQuestion: {question} [/INST] The answer is: "

    return processed_prompt




def qwen25_honest(data):
    return_list = []
    prompt_template = '''<|im_start|>system\nYou are a helpful assistant. Answer the question. If you don't know the answer to the question, it is appropriate to say "I don't know."\n<|im_end|>\n<|im_start|>user\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def qwen25_qa(data):
    prompt_template = '''<|im_start|>system\nPlease answer the question and provide the answer in the following format: "The answer is: ...".\n<|im_end|>\n<|im_start|>user\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt


def llama3_boundary_refine(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nEvaluate whether a given question can be answered based on your own knowledge. Categorize it into one of the following types:\n
                        Type 1: Confidently answerable based on your existing knowledge.\n
                        Type 2: Not answerable based on your existing knowledge. \n
                        Clearly state the type in this format: "The question type is: Type [Type Number]." Do not provide any explanation or answer to the question.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt



def llama3_boundary_last_token(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nEvaluate whether a given question can be answered based on your own knowledge. Categorize it into one of the following types:\n
                        Type 1: Confidently answerable based on your existing knowledge.\n
                        Type 2: Not answerable based on your existing knowledge. \n
                        Clearly state the type in this format: "The type is [Type Number]." Do not provide any explanation or answer to the question.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe Type is '''

    processed_prompt = prompt_template.format(question=data['question'])


    return processed_prompt



def llama3_reverse_qa_icl(data):
    prompt_template = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGenerate a one-sentence question with the answer: "{data['ground_truth']}". The only possible answer to the question must be "{data['ground_truth']}". The question should not contain the text "{data['ground_truth']}". Please format your output as "Question: [insert generated question]". If no possible question exists say "IDK".
    Below are some examples of how to generate the question:\n'''

    # 直接通过 data 生成例子部分
    examples = "\n".join(
        [f"Answer: {data[f'answer_icl{i}']}\nQuestion: {data[f'question_icl{i}']}" for i in range(1, 4)]
    )

    # 拼接最终的 prompt
    prompt = f"{prompt_template}{examples}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def deepseek_reverse_qa_icl(data):
    prompt_template = f'''Generate a one-sentence question with the answer: "{data['ground_truth']}". The only possible answer to the question must be "{data['ground_truth']}". The question should not contain the text "{data['ground_truth']}". Please format your output as "Question: [insert generated question]". If no possible question exists say "IDK".
    Below are some examples of how to generate the question:\n'''

    # 直接通过 data 生成例子部分
    examples = "\n".join(
        [f"Answer: {data[f'answer_icl{i}']}\nQuestion: {data[f'question_icl{i}']}" for i in range(1, 4)]
    )

    # 拼接最终的 prompt
    prompt = f"{prompt_template}{examples}\n"

    return prompt




def deepseek_check_fact(data):
    prompt_template = f'''You are tasked with evaluating the factuality of the information presented in the form of a set of question-answer pairs. Each pair consists of a specific question followed by an associated answer. 
    Analyze each answer carefully to determine whether it is factually accurate, inaccurate, or if there's insufficient information to make a definite judgment. 
    Consider established knowledge, historical facts, scientific principles, and common sense as benchmarks for your assessment. 
    Please clearly state for each pair whether the answer is "Factually Accurate", "Factually Inaccurate", or "Insufficient Information to Judge". And do not provide any explanation for your determination.
    Question: {data['question']}\nAnswer: {data['ground_truth']}\nFacutly: '''

    return prompt_template


def deepseek_check_answer(data):
    # 如果ground_truth是列表，则用逗号拼接；如果不是列表，则直接使用
    ground_truth = ', '.join(data['ground_truth']) if isinstance(data['ground_truth'], (list, tuple)) else data['ground_truth']
    
    prompt_template = f'''Given a question, ground truth, and answer, determine whether the answer is correct according to the ground truth. The answer can be a synonym or a more detailed or broader description of the ground truth, but it must not have a different meaning. Output "correct" if the answer is accurate, and "incorrect" if it is not. Do not provide any explanations.\n\n
    Question: {data['question']} \n
    Ground Truth: {ground_truth} \n
    Answer: {data['prediction']} \n
    '''

    return prompt_template





def qa_pair_prompt(data):
    prompt_template = f'''Question: {data['question']}\nAnswer: {data['ground_truth']}\n'''

    return prompt_template



def llama3_honest_plus_pair(data):
    """
    data: {'question': 'chosen': , 'reject': }
    """
    question = data['question']
    chosen = data['chosen']
    reject = data['reject']
    chosen_prompt = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{chosen}<|eot_id|>'''
    reject_prompt = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{reject}<|eot_id|>'''

    return [chosen_prompt, reject_prompt]


def gemma_honest_pair(data):
    """
    data: {'question': 'chosen': , 'reject': }
    """
    question = data['question']
    chosen = data['chosen']
    reject = data['reject']
    chosen_prompt = f'''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n{chosen}'''
    reject_prompt = f'''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n{reject}'''

    return [chosen_prompt, reject_prompt]


def mistral_honest_pair(data):
    """
    data: {'question': 'chosen': , 'reject': }
    """
    question = data['question']
    chosen = data['chosen']
    reject = data['reject']
    chosen_prompt = f'''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question} [/INST] {chosen}</s>'''
    reject_prompt = f'''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question} [/INST] {reject}</s>'''

    return [chosen_prompt, reject_prompt]


def mistral_honest_pair_test(data):
    """
    data: {
        "question": "",
        "ground_truth": "",
        "candidate_answers": [
            {
                "output": "",
                "idk": ,
                "score": ,
                "count": ,
                "conflict_label":
            }, ...
        ]
    }
    """
    return_list = []
    prompt_template = '''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question} [/INST] {answer}</s>'''

    for candidate in data['candidate_answers']:
        answer = candidate['output']
        filled_prompt = prompt_template.format(question=data['question'], answer=answer)
        return_list.append(filled_prompt)

    return return_list


def mistral_honest_list(data):
    return_list = []
    prompt_template = '''[INST] Answer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question} [/INST] {answer}</s>'''

    processed_prompt = prompt_template.format(question=data['question'], answer=data['output'])


    return processed_prompt


def gemma_honest_pair_test(data):
    """
    data: {
        "question": "",
        "ground_truth": "",
        "candidate_answers": [
            {
                "output": "",
                "idk": ,
                "score": ,
                "count": ,
                "conflict_label":
            }, ...
        ]
    }
    """
    return_list = []
    prompt_template = '''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n{answer}'''

    for candidate in data['candidate_answers']:
        answer = candidate['output']
        filled_prompt = prompt_template.format(question=data['question'], answer=answer)
        return_list.append(filled_prompt)
    
    return return_list

def gemma_honest_list(data):
    return_list = []
    prompt_template = '''<start_of_turn>user\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n{answer}'''

    processed_prompt = prompt_template.format(question=data['question'], answer=data['output'])


    return processed_prompt



def llama3_honest_plus_pair_test(data):
    """
    data: {
        "question": "",
        "ground_truth": "",
        "candidate_answers": [
            {
                "output": "",
                "idk": ,
                "score": ,
                "count": ,
                "conflict_label":
            }, ...
        ]
    }
    """
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>'''

    for candidate in data['candidate_answers']:
        answer = candidate['output']
        filled_prompt = prompt_template.format(question=data['question'], answer=answer)
        return_list.append(filled_prompt)
    
    return return_list



def llama3_honest_plus_rank_test(data):
    """
    data: {
        "question": "",
        "ground_truth": "",
        "candidate_answers": [
            {
                "output": "",
                "idk": ,
                "score": ,
                "count": ,
                "conflict_label":
            }, ...
        ]
    }
    """
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\"\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>'''

    for candidate in data['candidate_answers']:
        answer = candidate['output']
        filled_prompt = prompt_template.format(question=data['question'], answer=answer)
        return_list.append(filled_prompt)
    
    return return_list




def llama3_honest_plus_list(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\". Do not say \"I am not sure\" if you are sure about the answer.\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>'''

    processed_prompt = prompt_template.format(question=data['question'], answer=data['output'])


    return processed_prompt



def llama3_honest_plus_point(data):
    return_list = []
    prompt_template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question. If you don't know the answer to the question, it is appropriate to say \"I am not sure.\"\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>'''

    for candidate in data['candidate_answers']:
        answer = candidate['output']
        filled_prompt = prompt_template.format(question=data['question'], answer=answer)
        return_list.append(filled_prompt)
    
    return return_list
