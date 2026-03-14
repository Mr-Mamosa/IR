# 1. Define the Corpus
documents = {
    1: "BSc lectures start at 7",
    2: "My lectures are over",
    3: "Today is a holiday"
}

def build_index(docs):
    index = {}
    for doc_id, text in docs.items():
        terms = set(text.lower().split())
        for term in terms:
            if term not in index:
                index[term] = {doc_id}
            else:
                index[term].add(doc_id)
    return index

def boolean_not(operand, index, total_docs):
    operand_set = index.get(operand.lower(), set())
    all_docs_set = set(range(1, total_docs + 1))
    return list(all_docs_set.difference(operand_set))

inverted_index = build_index(documents)
query_term = "lectures"
result = boolean_not(query_term, inverted_index, len(documents))

print("Inverted Index:", inverted_index)
print(f"\nQuery: NOT {query_term}")
