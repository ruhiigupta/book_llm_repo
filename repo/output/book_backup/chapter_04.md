# Chapter 4: Token Embeddings and Positional Encoding

## Understanding Token Embeddings

Token embeddings are a crucial component of large language models, as they enable the model to capture the semantic meaning of individual tokens. However, a major limitation of token embeddings is that they do not inherently convey positional information. This can lead to difficulties in tasks that rely on the order of tokens, such as understanding the nuances of sentence structure or capturing the context of a particular word.

To address this limitation, absolute positional encoding is employed, where a unique embedding vector is added to the token embedding for each position in the input sequence. This positional embedding vector is added to the token embedding, effectively modifying the representation of the token based on its position. For example, consider a sequence of tokens: "cat," "sat," "on," and "the." If the token embedding vector for all four tokens is similar, such as [1, 1, 1], the absolute positional encoding would add a unique positional embedding vector to each token, reflecting its position in the sequence.

## Importance of Positional Embeddings

Positional information is essential for large language models to accurately capture the relationships between tokens in a sequence. Without it, models may struggle to understand the nuances of language, such as the difference between "cat sat on the mat" and "mat sat on the cat." The main drawback of just token embeddings is that they do not inherently include positional information.

In absolute positional encoding, a unique embedding vector is added to the token embedding for each position in the input sequence. This allows the model to differentiate between tokens based on their position. For example, if we have the tokens "cat," "sat," "on," and "the," the token embedding vectors for all of these tokens may be similar, such as [1, 1, 1,...]. However, in absolute positional encoding, another embedding vector is added to the token embedding, which is the positional embedding vector. Since these words have different positions, their positional embedding vectors will be distinct, allowing the model to capture the positional relationships between tokens.

For instance, in the sequence "cat sat on the mat," the positional embedding vectors for the tokens "cat" and "sat" will be different, reflecting their positions in the sequence. This allows the model to better understand the relationships between tokens and improve its overall performance.