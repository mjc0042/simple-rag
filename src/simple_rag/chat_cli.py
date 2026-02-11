class ChatCLI:

    def __init__(self, rag_engine):
        self.rag = rag_engine

    def run(self):
        print("RAG Chat started. Type 'exit' to quit.\n")

        while True:
            question = input(">> ")
            if question.lower() in {"exit", "quit"}:
                break

            answer = self.rag.ask(question)
            print("\n" + answer + "\n")
