package simpletokenizer

import (
	"encoding/json"
	"os"
	"strings"
)

type Tokenizer struct {
	Vocab       map[string]int
	IndexToWord []string
}

// NewTokenizer создаёт новый токенизатор
func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Vocab:       make(map[string]int),
		IndexToWord: []string{},
	}
}

// BuildVocabulary строит словарь на основе символов в текстах
func (t *Tokenizer) BuildVocabulary(texts []string) {
	uniqueChars := make(map[string]struct{})
	for _, text := range texts {
		for _, char := range text {
			charStr := string(char)
			if _, exists := uniqueChars[charStr]; !exists {
				uniqueChars[charStr] = struct{}{}
			}
		}
	}

	for char := range uniqueChars {
		t.Vocab[char] = len(t.IndexToWord)
		t.IndexToWord = append(t.IndexToWord, char)
	}
}

// Tokenize преобразует текст в последовательность индексов символов
func (t *Tokenizer) Tokenize(text string) []int {
	tokens := make([]int, 0, len(text))
	for _, char := range text {
		charStr := string(char)
		if idx, exists := t.Vocab[charStr]; exists {
			tokens = append(tokens, idx)
		} else {
			// Если символ не найден, добавляем специальный токен
			tokens = append(tokens, -1) // -1 для неизвестных символов
		}
	}
	return tokens
}

// Decode преобразует последовательность индексов обратно в текст
func (t *Tokenizer) Decode(tokens []int) string {
	words := make([]string, len(tokens))
	for i, token := range tokens {
		if token >= 0 && token < len(t.IndexToWord) {
			words[i] = t.IndexToWord[token]
		} else {
			words[i] = "<UNK>" // UNK для неизвестных слов
		}
	}
	return strings.Join(words, " ")
}

// SaveVocabulary сохраняет словарь в JSON файл
func (t *Tokenizer) SaveVocabulary(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(t)
}

// LoadVocabulary загружает словарь из JSON файла
func (t *Tokenizer) LoadVocabulary(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	return decoder.Decode(t)
}
