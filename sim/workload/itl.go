package workload

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ITLRecord represents one chunk timestamp in the ITL trace.
// ITL traces capture per-chunk arrival times during streaming inference
// for Inter-Token Latency (ITL) calibration.
type ITLRecord struct {
	RequestID   int   // Request identifier (matches TraceRecord.RequestID)
	ChunkIndex  int   // Chunk sequence number (0 = first chunk / TTFT)
	TimestampUs int64 // Absolute timestamp in microseconds (UnixMicro)
}

// ExportITL writes ITL records to a CSV file.
// Format: request_id,chunk_index,timestamp_us
// Timestamps use integer formatting to preserve microsecond precision.
func ExportITL(records []ITLRecord, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating ITL file: %w", err)
	}
	defer func() { _ = file.Close() }()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"request_id", "chunk_index", "timestamp_us"}); err != nil {
		return fmt.Errorf("writing ITL CSV header: %w", err)
	}

	// Write data rows
	for _, r := range records {
		row := []string{
			strconv.Itoa(r.RequestID),
			strconv.Itoa(r.ChunkIndex),
			strconv.FormatInt(r.TimestampUs, 10),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("writing ITL CSV row (request_id=%d): %w", r.RequestID, err)
		}
	}
	return nil
}

// LoadITL reads ITL records from a CSV file.
// Validates that all fields are non-negative (R3, R20).
func LoadITL(path string) ([]ITLRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening ITL file: %w", err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	// Skip header row
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("reading ITL CSV header: %w", err)
	}

	var records []ITLRecord
	rowNum := 1 // 0 = header
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading ITL CSV row %d: %w", rowNum, err)
		}
		rowNum++

		if len(row) < 3 {
			return nil, fmt.Errorf("ITL CSV row %d has %d columns, expected 3", rowNum, len(row))
		}

		requestID, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, fmt.Errorf("parsing request_id at row %d: %w", rowNum, err)
		}
		chunkIndex, err := strconv.Atoi(row[1])
		if err != nil {
			return nil, fmt.Errorf("parsing chunk_index at row %d: %w", rowNum, err)
		}
		timestampUs, err := strconv.ParseInt(row[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parsing timestamp_us at row %d: %w", rowNum, err)
		}

		// Validate: no negative values (R3, R20)
		if requestID < 0 || chunkIndex < 0 || timestampUs < 0 {
			return nil, fmt.Errorf("ITL CSV row %d has negative value (request_id=%d, chunk_index=%d, timestamp_us=%d)", rowNum, requestID, chunkIndex, timestampUs)
		}

		records = append(records, ITLRecord{
			RequestID:   requestID,
			ChunkIndex:  chunkIndex,
			TimestampUs: timestampUs,
		})
	}
	return records, nil
}
