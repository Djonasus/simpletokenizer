package simpletokenizer

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Embedding struct {
	VocabSize    int        // Размер словаря
	EmbeddingDim int        // Размерность встраиваний
	Weights      *mat.Dense // Матрица встраиваний (VocabSize x EmbeddingDim)
}

// NewEmbedding создает слой встраивания
func NewEmbedding(vocabSize, embeddingDim int) *Embedding {
	e := &Embedding{
		VocabSize:    vocabSize,
		EmbeddingDim: embeddingDim,
		Weights:      mat.NewDense(vocabSize, embeddingDim, nil),
	}

	// Инициализируем веса случайными значениями
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embeddingDim; j++ {
			e.Weights.Set(i, j, rand.NormFloat64()*0.01)
		}
	}

	return e
}

// Forward преобразует токены в вектора встраиваний
func (e *Embedding) Forward(tokens []int) []*mat.VecDense {
	vectors := make([]*mat.VecDense, len(tokens))
	for i, token := range tokens {
		if token >= 0 && token < e.VocabSize {
			vector := mat.NewVecDense(e.EmbeddingDim, nil)
			for j := 0; j < e.EmbeddingDim; j++ {
				vector.SetVec(j, e.Weights.At(token, j))
			}
			vectors[i] = vector
		} else {
			// Для неизвестных токенов возвращаем нулевой вектор
			vectors[i] = mat.NewVecDense(e.EmbeddingDim, nil)
		}
	}
	return vectors
}

// DecodeVector преобразует вектор обратно в токен
func (e *Embedding) DecodeVector(vector *mat.VecDense) int {
	bestToken := -1
	bestScore := -1.0

	// Нормализуем входной вектор (для косинусного сходства)
	vectorNorm := mat.Norm(vector, 2)
	normalizedVector := mat.NewVecDense(e.EmbeddingDim, nil)
	for i := 0; i < e.EmbeddingDim; i++ {
		normalizedVector.SetVec(i, vector.AtVec(i)/vectorNorm)
	}

	// Сравниваем с каждым вектором в матрице встраиваний
	for token := 0; token < e.VocabSize; token++ {
		row := e.Weights.RowView(token)

		// Нормализуем вектор из матрицы встраиваний
		rowNorm := mat.Norm(row, 2)
		normalizedRow := mat.NewVecDense(e.EmbeddingDim, nil)
		for j := 0; j < e.EmbeddingDim; j++ {
			normalizedRow.SetVec(j, row.AtVec(j)/rowNorm)
		}

		// Вычисляем косинусное сходство
		score := mat.Dot(normalizedVector, normalizedRow)
		if score > bestScore {
			bestScore = score
			bestToken = token
		}
	}

	return bestToken
}

func (e *Embedding) Decode(vectors []*mat.VecDense) []int {
	tokens := make([]int, len(vectors))

	for i, vector := range vectors {
		tokens[i] = e.DecodeVector(vector)
	}

	return tokens
}
