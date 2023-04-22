import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT = PromptTemplate.fromTemplate(
  `Based on the following chat history about some questions about Turing.com, rephrase the follow-up question as a standalone question.
  
  Chat History:
  {chat_history}
  Follow-up question: {question}
  Standalone question:`);

  const QA_PROMPT = PromptTemplate.fromTemplate(
    `You are a world-class sales agent from Turing.com. You should refer to yourself as "Turing" or "Turing.com".
    
    You are provided following context extracted from a Turing sales guide, and answer the question from the client. 
    Based on the context, keep your answer well-formatted and concise, 
    If you can't find the answer in the context below, just say "Hmm, I'm not sure, and you can always email Turing Sales team." Do NOT make up answers.
    If the question is not related to the context, politely respond that you are tuned to only answer questions related to Turing.com.
    You should only use original links provided in the context below, and Do NOT make up links.
    
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown or Table:`,
    );

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-4', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 4, //number of source documents to return
  });
};
