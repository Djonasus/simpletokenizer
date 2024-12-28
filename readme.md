# Простая библиотека токенизации и векторного преобразования

используется в связке с NNSimp. Реализует посимвольную токенизацию. Работа с библиотекой крайне простая:

```go
func main() {
    tokenizer := simpletokenizer.NewTokenizer()
    texts := []string{"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz", "! .,-:", "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЧШЦЩЭЬЪЮЯ", "абвгдеёжзийклмнопрстуфхчшцщэьъюя", "1234567890-+=/*"}
	tokenizer.BuildVocabulary(texts)

    // Для токенизации
    tokens := tokenizer.Tokenize("2 * 2 = 4. Это всем известно в целом мире!")

    // Для преобразования в текст
    original_text := tokenizer.Decode(tokens)

    // Для создания векторов из токенов
    vocabSize := len(tokenizer.Vocab)
	embeddingDim := 64
	embedding := simpletokenizer.NewEmbedding(vocabSize, embeddingDim)

    vectors := embedding.Forward(tokens)

    // Для обратного преобразования из векторов в токены
    original_tokens = embedding.Decode(tokens)

    // Для сохранения словаря (не эмбеддинга)
    err := tokenizer.SaveVocabulary("vocab.json")
	if err != nil {
		panic(err)
	}

    // Для загрузки
    err := tokenizer.LoadVocabulary("vocab.json")
	if err != nil {
		panic(err)
	}
}
```
