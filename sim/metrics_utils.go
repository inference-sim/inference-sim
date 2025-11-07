// sim/metrics_utils.go
package sim

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

// Bin represents a single completion time integral bin with its integer key and count.
type Bin struct {
	Key   int
	Count int
}

// CalculatePercentile is a util function that calculates the p-th percentile of a data list
func CalculatePercentile(data []float64, p float64) float64 {
	n := len(data)

	sortedData := make([]float64, n)
	copy(sortedData, data)

	sort.Float64s(sortedData)

	rank := p / 100.0 * float64(n-1)
	lowerIdx := int(math.Floor(rank))
	upperIdx := int(math.Ceil(rank))

	if lowerIdx == upperIdx {
		return sortedData[lowerIdx]
	} else {
		lowerVal := sortedData[lowerIdx]
		upperVal := sortedData[upperIdx]
		if upperIdx >= n {
			return sortedData[n-1]
		}
		return lowerVal + (upperVal-lowerVal)*(rank-float64(lowerIdx))
	}
}

func (m *Metrics) SavetoFile(data []int, fileName string) {
	file, err := os.OpenFile(fileName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		logrus.Fatalf("Error creating file %s: %v\n", fileName, err)
		return
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			logrus.Fatalf("Error closing file %s: %v\n", fileName, closeErr)
		}
	}()

	writer := bufio.NewWriter(file)

	defer func() {
		if flushErr := writer.Flush(); flushErr != nil {
			logrus.Fatalf("Error flushing writer for file %s: %v\n", fileName, flushErr)
		}
	}()

	for _, f := range data {
		_, writeErr := fmt.Fprint(writer, f, ", ")
		if writeErr != nil {
			logrus.Fatalf("Error writing int %d to file: %v\n", f, writeErr)
			return // Stop writing on first error
		}
	}

	logrus.Debugf("Successfully wrote to '%s'\n", fileName)
}

// keyWithIndex is a util struct to temporarily store the original key and its parsed numeric index.
// useful for metrics parsing when requestIDs are strings of form: "request_1", "request_2" ...
type keyWithIndex struct {
	originalKey  string
	numericIndex int
}

// SortRequestMetrics sorts a map of "request_N": value pairs based on the numerical part of the request key N
// This is useful for metric maps like {"request_N": TTFT/TPOT/CompletionTime}
// It returns a slice of RequestPair structs, sorted numerically by their Index
func SortRequestMetrics(data map[string]float64) []float64 {
	if len(data) == 0 {
		return []float64{}
	}

	keysWithIndexes := make([]keyWithIndex, 0, len(data))
	for key := range data {
		trimmedKey := strings.TrimPrefix(key, "request_")
		if trimmedKey == key && !strings.HasPrefix(key, "request_") {
			return nil
		}

		numericIndex, _ := strconv.Atoi(trimmedKey)

		keysWithIndexes = append(keysWithIndexes, keyWithIndex{
			originalKey:  key,
			numericIndex: numericIndex,
		})
	}

	sort.Slice(keysWithIndexes, func(i, j int) bool {
		return keysWithIndexes[i].numericIndex < keysWithIndexes[j].numericIndex
	})

	sortedValues := make([]float64, 0, len(keysWithIndexes))
	for _, ki := range keysWithIndexes {
		sortedValues = append(sortedValues, data[ki.originalKey])
	}

	return sortedValues
}
