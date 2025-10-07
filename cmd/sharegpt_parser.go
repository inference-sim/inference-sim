package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// Prompt represents a single request sent to vLLM.
type Prompt struct {
	inputText     []int   `json:"input_text"`     // must be tokenized
	generatedText []int   `json:"generated_text"` // must be tokenized
	inputLen      int     `json:"input_len"`
	prefixLen     int     `json:"prefix_len"`
	outputLen     int     `json:"output_len"`
	e2eLatency    float64 `json:"e2e_latency"`
	arrivalTime   int64   `json:"arrival_time"`
}

// DataEntry represents a single top-level object in the JSON file.
type DataEntry struct {
	numPrompts int      `json:"num_prompts"`
	rate       float64  `json:"request_rate"`
	prompts    []Prompt `json:"prompts"`
}

// Process input tokenized requests JSON file to extract only the first human-gpt from-value pair
// from conversations
func ProcessInputShareGPT(requestsFilePath string) []*sim.Request {

	// Read the content of the JSON file
	fileContent, err := os.ReadFile(requestsFilePath)
	if err != nil {
		logrus.Fatalf("Error reading file %s: %v", requestsFilePath, err)
	}

	// Declare a slice of DataEntry to unmarshal the raw JSON into
	var rawData []DataEntry

	// Unmarshal the JSON data into the rawData slice
	err = json.Unmarshal(fileContent, &rawData)
	if err != nil {
		logrus.Fatalf("Error unmarshaling raw JSON: %v", err)
	}

	var requests []*sim.Request

	for _, entry := range rawData {
		// We need to find the first human-gpt pair
		var inputTokens []int
		var outputTokens []int

		for i := 0; i < len(entry.prompts)-1; i++ {
			reqID := fmt.Sprintf("request{%d}", i)
			inputTokens = entry.prompts[i].inputText
			outputTokens = entry.prompts[i+1].generatedText

			req := sim.Request{
				ID:           reqID,
				InputTokens:  inputTokens,
				OutputTokens: outputTokens,
				ArrivalTime:  entry.prompts[i].arrivalTime,
			}
			requests = append(requests, &req)
		}
	}

	// Print the parsed requests to verify
	logrus.Infof("Successfully parsed requests file %s. Total number of requests: %d\n", requestsFilePath, len(requests))
	return requests
}
