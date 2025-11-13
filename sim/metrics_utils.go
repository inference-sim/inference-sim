// sim/metrics_utils.go
package sim

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"

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

type IntOrFloat64 interface {
	int | float64
}

func CalculateMean[T IntOrFloat64](numbers []T) float64 {
	if len(numbers) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, number := range numbers {
		sum += float64(number)
	}

	return sum / float64(len(numbers))
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
