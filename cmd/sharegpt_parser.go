package cmd

import (
	"encoding/json"
	"os"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// Conversation represents a single chat turn in the conversation.
type Conversation struct {
	From  string `json:"from"`
	Value []int  `json:"value"`
}

// DataEntry represents a single top-level object in your JSON array.
type DataEntry struct {
	ID            string         `json:"id"`
	Conversations []Conversation `json:"conversations"`
	ArrivalDelta  int            `json:"arrivalDelta"`
}

// Process input tokenized requests JSON file to extract only the first human-gpt from-value pair
// from conversations
func ProcessInput(requestsFilePath string) []*sim.Request {

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
		pairCount := 0

		for i := 0; i < len(entry.Conversations)-1 && pairCount < 2; i++ {
			if entry.Conversations[i].From == "human" && entry.Conversations[i+1].From == "gpt" {
				inputTokens = entry.Conversations[i].Value
				outputTokens = entry.Conversations[i+1].Value

				req := sim.Request{
					ID:           entry.ID,
					InputTokens:  inputTokens,
					OutputTokens: outputTokens,
					ArrivalDelta: entry.ArrivalDelta,
				}
				requests = append(requests, &req)
				pairCount++
				i++ // Skip the gpt conversation as it's already processed
			}
		}
	}

	// Print the parsed requests to verify
	logrus.Infof("Successfully parsed requests file %s. Total number of requests: %d\n", requestsFilePath, len(requests))
	return requests
}
